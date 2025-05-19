import boto3
import os
import sys
import json
import time
import uuid
from datetime import datetime
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import cv2
import io
from io import BytesIO

# Configuraciones globales
DESTINATION_BUCKET = "silver-honne-sep"
STEP_FUNCTION_ARN = "arn:aws:states:us-east-1:153788051293:stateMachine:test_ec2"

# Inicializa clientes AWS
s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
stepfunctions = boto3.client("stepfunctions", region_name="us-east-1")

# UUID para esta ejecución
execution_id = str(uuid.uuid4())
start_time = time.time()
start_dt = datetime.utcnow()

# Leer mensaje de evento
with open(sys.argv[1], 'r') as f:
    msg = json.load(f)

bucket = msg["detail"]["bucket"]["name"]
key = msg["detail"]["object"]["key"]
event_id = msg["id"]
filename = key.split("/")[-1]
local_pdf_path = f"/tmp/{filename}"
output_pdf_path = f"/tmp/processed_{filename}"
dest_key = f"preprocesado/{os.path.basename(key)}"

print(f"[{execution_id}] Iniciando procesamiento para {key}")

# Descargar archivo desde S3
s3.download_file(bucket, key, local_pdf_path)
print(f"[{execution_id}] PDF descargado desde S3")

# Clasificación documental con Bedrock
def clasificacion_documentos(bucket, key):
    file_extension = key.split('.')[-1].lower()
    response = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = response['Body'].read()

    messages = [{
        "role": "user",
        "content": [
            {
                "document": {
                    "format": file_extension,
                    "name": f"Documento_{file_extension}",
                    "source": {"bytes": file_bytes}
                }
            },
            {
                "text": "Eres un experto en clasificar documentos oficiales mexicanos.\n" + \
                        "Clasifícalo, de acuerdo a su contenido y características visuales, en alguno de los siguientes tipos de documento:\n" + \
                        "- \"INE_IFE\"; Credencial para votar de mexico\n" + \
                        "- \"Poder_Notarial\"; Contiene un encabezado de identificacion del notario ,ademas poliza del libro notarial y Firma del Notario/Corredor Público y Sello.\n" + \
                        "- \"Acta_Constitutiva\"; Aunque el documento en sí es una Escritura Pública, Instrumento Notarial o Póliza, su contenido se referiere explícitamente a la Constitución de la Sociedad Mercantil o la formalización de actos relacionados con la vida de una sociedad, es indispensable la participación de un Notario Público o Corredor Público, quien HACE CONSTAR  el acto, los datos del Notario Público o Corredor Público (nombre, número, lugar de adscripción) y sellos son prominentes, se identifican claramente las personas físicas o morales que constituyen la sociedad.\n" + \
                        "- \"Constancia_de_situación_fiscal\";Documento oficial emitido por el SAT que incluye el logotipo del SAT, datos de identificación del contribuyente (nombre, RFC, CURP), domicilio fiscal, datos de ubicación, actividades económicas, regímenes, obligaciones, y fecha de inicio de operaciones. Suele tener un código QR y una leyenda que indica su validez oficial.\n" + \
                        "- \"Declaración_Anual_del_SAT\"; Contiene frases explicitas como DECLARACIÓN DEL EJERCICIO DE IMPUESTOS FEDERALES, tiene presencia de los logotipos o menciones de HACIENDA y SAT (Servicio de Administración Tributaria), cuenta con la Indicación clara del año fiscal al que corresponde la declaración, por ejemplo, Ejercicio: 2023 .\n" + \
                        "- \"Opiniones_de_cumplimiento\" (SAT, IMSS, INFONAVIT).\n" + \
                        "- \"Opinión del cumplimiento de obligaciones fiscales en materia de Seguridad Social\"; Este archivo lo crea la institución llamada Instituto Mexicano del Seguro Social (IMSS).\n" + \
                        "- \"Opinión del cumplimiento de obligaciones fiscales\"; Servicio de Administración Tributaria. Este archivo lo crea la institución llamada SAT.\n" + \
                        "- \"Opinión del cumplimiento de INFONAVIT\"; Usualmente el documento tiene como título: Constancia de Situación Fiscal en materia obligaciones Fiscales relativa a las aportaciones patronales y entero de descuentos. Este archivo lo crea la institución llamada INFONAVIT.\n" + \
                        "- \"Documento desconocido\"; Cuando no se tiene la certeza del tipo de documento o no entra en los tipos anteriomente descritos.\n" + \
                        "- \"Estado_de_cuenta\"; Documento bancario que muestra los movimientos y saldo de una cuenta. Generalmente incluye el nombre del banco, número de cuenta, nombre del titular, periodo del estado de cuenta, lista de transacciones con fechas y montos, y saldo inicial y final.\n" + \
                        "- \"Comprobante_de_domicilio\"; el cual puede ser de agua, luz, teléfono, entre otros. en ellos contienen informacion del total a pagar del servicio.\n" + \
                        "- \"Cedula_Profesional\".\n" + \
                        "- \"Curriculum_vitae\"; Este archivo a veces tiene la palabra Curriculum. Este archivo contiene el perfil de una empresa. Usualmente contiene información sobre la historia, la misión, los servicios que ofrecen y los proyectos realizados por la empresa. Usualmente incluyen la información de contacto de los representantes de la empresa.\n" + \
                        "- \"Carta_Responsiva\".\n" + \
                        "Devuelve la respuesta en formato JSON, con la clave principal \"tipoDocumento\", solamentamente el tipo documento (nada de descripcion que van despues del punto y coma).\n" + \
                        "Por favor y muchas gracias."
            }
        ]
    }]

    inf_config = {"maxTokens": 5000, "temperature": 0.1}

    response = bedrock.converse(
        modelId="us.amazon.nova-pro-v1:0", 
        messages=messages,
        inferenceConfig=inf_config
    )
    return response['output']['message']['content'][0]['text']

# Procesamiento de imágenes con OpenCV
def process_image_opencv(image_np, max_rois=2):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Detección de bordes y dilatación
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ordenar todos los contornos por área (de mayor a menor)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Limitar a los max_rois contornos más grandes
    contours = contours[:max_rois]
    
    # Lista para almacenar las ROIs
    all_rois = []
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calcular las nuevas dimensiones ampliadas (factor 1.5)
        center_x, center_y = x + w/2, y + h/2
        new_w, new_h = int(w * 1.5), int(h * 1.5)
        
        # Calcular las nuevas coordenadas manteniendo el centro
        new_x = max(0, int(center_x - new_w/2))
        new_y = max(0, int(center_y - new_h/2))
        
        # Asegurarse de que el ROI ampliado no exceda los límites de la imagen
        new_w = min(image_np.shape[1] - new_x, new_w)
        new_h = min(image_np.shape[0] - new_y, new_h)
        
        # Extraer el ROI ampliado
        roi = gray[new_y:new_y+new_h, new_x:new_x+new_w]
        
        # Guardar solo el ROI
        all_rois.append(roi)
    
    return all_rois

    
# EXTRAER IMAGENES Y PROCESAR
def extract_and_process_images(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    numero_de_paginas = doc.page_count
    processed_images = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)
        
        # Procesar la imagen y obtener las 2 ROIs más grandes
        rois = process_image_opencv(img_np, max_rois=2)
        
        # Convertir cada ROI a PIL Image y agregarlos a la lista
        pil_rois = []
        for roi in rois:
            # Convertir cada ROI de numpy array a PIL Image
            pil_roi = Image.fromarray(roi)
            # Asegurarse de que sea RGB si es necesario
            if pil_roi.mode != 'RGB':
                pil_roi = pil_roi.convert('RGB')
            pil_rois.append(pil_roi)
        
        processed_images.append(pil_rois)
    
    doc.close()
    return processed_images, numero_de_paginas

def save_rois_to_s3_one_per_page(imagenes, dpi=150, page_size="A4", region='us-east-1'):
    # Tamaños de página en puntos
    PAGE_SIZES = {
        "A4": (595, 842),
        "Letter": (612, 792),
    }
    page_width, page_height = PAGE_SIZES.get(page_size, PAGE_SIZES["A4"])

    # Crear PDF en memoria
    doc = fitz.open()

    for rois in imagenes:
        for roi in rois:
            if roi.mode != 'RGB':
                roi = roi.convert('RGB')

            # 🔍 Escalar la imagen a 1.5x su tamaño original
            roi = roi.resize((int(roi.width * 1.5), int(roi.height * 1.5)))

            # Calcular dimensiones del ROI en puntos
            roi_width_pt = roi.width * 72 / dpi
            roi_height_pt = roi.height * 72 / dpi

            # Escalar para que quepa en la página
            scale_factor = min(page_width / roi_width_pt, page_height / roi_height_pt, 1.0)
            roi_width_pt *= scale_factor
            roi_height_pt *= scale_factor

            resized_roi = roi.resize((int(roi_width_pt * dpi / 72), int(roi_height_pt * dpi / 72)))

            # Guardar la imagen como PNG en memoria
            img_buffer = BytesIO()
            resized_roi.save(img_buffer, format="PNG")

            # Crear nueva página
            page = doc.new_page(width=page_width, height=page_height)

            # Insertar imagen centrada
            x0 = (page_width - roi_width_pt) / 2
            y0 = (page_height - roi_height_pt) / 2
            rect = fitz.Rect(x0, y0, x0 + roi_width_pt, y0 + roi_height_pt)
            page.insert_image(rect, stream=img_buffer.getvalue())

    # Guardar el PDF en memoria
    pdf_buffer = BytesIO()
    doc.save(pdf_buffer)
    doc.close()

    # Subir a S3
    s3.put_object(
                Bucket=DESTINATION_BUCKET,
                Key=dest_key,
                Body=pdf_buffer.getvalue(),
                ContentType='application/pdf'
            )

    print(f"✅ PDF subido a s3://{DESTINATION_BUCKET}/{key}")

# Procesar PDF

# Guardar imágenes como un solo PDF
""" pdf_bytes = io.BytesIO()
if imagenes:
    imagenes[0].save(pdf_bytes, format="PDF", save_all=True, append_images=imagenes[1:])
    pdf_bytes.seek(0)

    output_key = f"preprocesado/{filename}"
    s3.upload_fileobj(pdf_bytes, DESTINATION_BUCKET, output_key)
    print(f"[{execution_id}] PDF procesado y subido a: {output_key}")
else:
    print(f"[{execution_id}] No se generaron imágenes.") """

# Clasificar
try:
    tipo_doc_str = clasificacion_documentos(bucket, key)
    tipo_doc = json.loads(tipo_doc_str)
except Exception as e:
    print(f"[{execution_id}] Error al clasificar: {e}")
    tipo_doc = {"tipoDocumento": "Documento desconocido"}

if "tipoDocumento" in tipo_doc:
    print(f"TIPO:{tipo_doc}")
    if "INE_IFE" in tipo_doc["tipoDocumento"]:
        imagenes, numero_de_paginas = extract_and_process_images(local_pdf_path)
        save_rois_to_s3_one_per_page(imagenes)
    else:
        docu = fitz.open(local_pdf_path)
        # Guardar en memoria y subir
        numero_de_paginas = docu.page_count
        pdf_buffer = io.BytesIO()
        docu.save(pdf_buffer)
        pdf_buffer.seek(0)
        s3.put_object(
                Bucket=DESTINATION_BUCKET,
                Key=dest_key,
                Body=pdf_buffer,
                ContentType='application/pdf'
            )

        print(f"✅ PDF subido a s3://{DESTINATION_BUCKET}/{key}")

# Lanzar Step Function
input_sf = {
    "Bucket": DESTINATION_BUCKET,
    "Key": dest_key,
    "Tipo": tipo_doc,
    "uuid": event_id,
    "Num_pags":numero_de_paginas
}

try:
    response = stepfunctions.start_execution(
        stateMachineArn=STEP_FUNCTION_ARN,
        name=f"exec-{execution_id}",
        input=json.dumps(input_sf)
    )
    print(f"[{execution_id}] Step Function lanzada correctamente.")
except Exception as e:
    print(f"[{execution_id}] Error al lanzar Step Function: {e}")

# Final
end_dt = datetime.utcnow()
duration = round(time.time() - start_time, 2)
print(f"[{execution_id}] Proceso terminado en {duration} segundos. Fin: {end_dt} UTC")
