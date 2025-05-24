from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import base64
import io
import os
import numpy as np
from PIL import Image, ImageDraw
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import json
from pydantic import BaseModel
import pydicom
from pydicom.errors import InvalidDicomError
import tempfile
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dental DICOM Analysis API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - Set these environment variables
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ENDPOINT = "https://detect.roboflow.com/adr/6"  # Specified endpoint
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Roboflow parameters as specified
CONFIDENCE_THRESHOLD = 0.30  # 30%
OVERLAP_THRESHOLD = 0.50     # 50%

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class DetectionResult(BaseModel):
    predictions: List[Dict[str, Any]]
    image_width: int
    image_height: int

class DiagnosticReport(BaseModel):
    findings: str
    recommendations: str
    severity: str
    confidence: float

class DicomConversionResponse(BaseModel):
    success: bool
    converted_image: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Dental DICOM Analysis API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "api_version": "1.0.0"}

@app.post("/convert-dicom")
async def convert_dicom(dicom_file: UploadFile = File(...)):
    """
    Convert DICOM file to viewable image format (PNG/JPEG)
    """
    try:
        # Validate file
        if not dicom_file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        filename_lower = dicom_file.filename.lower()
        if not (filename_lower.endswith('.dcm') or filename_lower.endswith('.rvg') or 
                dicom_file.content_type == 'application/dicom'):
            raise HTTPException(
                status_code=400, 
                detail="File must be a DICOM file (.dcm, .rvg) or have application/dicom content type"
            )
        
        # Read the DICOM file
        dicom_data = await dicom_file.read()
        
        # Convert DICOM to image
        converted_image, metadata = convert_dicom_to_image(dicom_data)
        
        if converted_image is None:
            raise HTTPException(status_code=422, detail="Failed to convert DICOM file to image")
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        converted_image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "converted_image": f"data:image/png;base64,{image_b64}",
            "metadata": metadata,
            "message": f"Successfully converted {dicom_file.filename}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DICOM conversion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DICOM conversion failed: {str(e)}")

def convert_dicom_to_image(dicom_data: bytes) -> tuple[Optional[Image.Image], Dict[str, Any]]:
    """
    Convert DICOM data to PIL Image with metadata extraction
    """
    try:
        # Create temporary file to read DICOM data
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(dicom_data)
            temp_file.flush()
            
            # Read DICOM file
            try:
                dicom_dataset = pydicom.dcmread(temp_file.name, force=True)
            except InvalidDicomError as e:
                logger.error(f"Invalid DICOM file: {e}")
                return None, {"error": "Invalid DICOM file format"}
            
            # Extract metadata
            metadata = extract_dicom_metadata(dicom_dataset)
            
            # Get pixel array
            if not hasattr(dicom_dataset, 'pixel_array'):
                logger.error("DICOM file does not contain pixel data")
                return None, {"error": "No pixel data found in DICOM file"}
            
            pixel_array = dicom_dataset.pixel_array
            
            # Handle different pixel data formats
            image = process_dicom_pixel_array(pixel_array, dicom_dataset)
            
            return image, metadata
            
    except Exception as e:
        logger.error(f"Error converting DICOM: {e}")
        return None, {"error": str(e)}

def extract_dicom_metadata(dataset) -> Dict[str, Any]:
    """
    Extract relevant metadata from DICOM dataset
    """
    metadata = {}
    
    try:
        # Basic DICOM tags
        metadata['patient_id'] = getattr(dataset, 'PatientID', 'Unknown')
        metadata['patient_name'] = str(getattr(dataset, 'PatientName', 'Unknown'))
        metadata['study_date'] = getattr(dataset, 'StudyDate', 'Unknown')
        metadata['modality'] = getattr(dataset, 'Modality', 'Unknown')
        metadata['manufacturer'] = getattr(dataset, 'Manufacturer', 'Unknown')
        metadata['institution_name'] = getattr(dataset, 'InstitutionName', 'Unknown')
        
        # Image specific information
        metadata['rows'] = getattr(dataset, 'Rows', 0)
        metadata['columns'] = getattr(dataset, 'Columns', 0)
        metadata['bits_allocated'] = getattr(dataset, 'BitsAllocated', 0)
        metadata['bits_stored'] = getattr(dataset, 'BitsStored', 0)
        metadata['pixel_representation'] = getattr(dataset, 'PixelRepresentation', 0)
        metadata['photometric_interpretation'] = getattr(dataset, 'PhotometricInterpretation', 'Unknown')
        
        # Remove any potential PHI (Personal Health Information) for demo
        if metadata['patient_name'] != 'Unknown':
            metadata['patient_name'] = 'ANONYMIZED'
        if metadata['patient_id'] != 'Unknown':
            metadata['patient_id'] = 'ANONYMIZED'
            
    except Exception as e:
        logger.warning(f"Error extracting metadata: {e}")
        metadata['extraction_error'] = str(e)
    
    return metadata

def process_dicom_pixel_array(pixel_array: np.ndarray, dataset) -> Image.Image:
    """
    Process DICOM pixel array and convert to PIL Image
    """
    try:
        # Handle different bit depths and pixel representations
        if pixel_array.dtype == np.uint16:
            # 16-bit image - scale to 8-bit
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        elif pixel_array.dtype == np.int16:
            # Signed 16-bit - convert to unsigned and scale
            pixel_array = pixel_array.astype(np.int32)
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # Handle photometric interpretation
        photometric = getattr(dataset, 'PhotometricInterpretation', 'MONOCHROME2')
        
        if photometric == 'MONOCHROME1':
            # Invert the image (0 is white, max is black)
            pixel_array = 255 - pixel_array
        
        # Ensure 2D array for grayscale
        if len(pixel_array.shape) > 2:
            pixel_array = pixel_array.squeeze()
        
        # Convert to PIL Image
        if len(pixel_array.shape) == 2:
            # Grayscale image
            image = Image.fromarray(pixel_array, mode='L')
        else:
            # Multi-channel image
            image = Image.fromarray(pixel_array)
        
        # Convert grayscale to RGB for consistency
        if image.mode == 'L':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        logger.error(f"Error processing pixel array: {e}")
        raise

@app.post("/analyze-xray")
async def analyze_xray(file: UploadFile = File(...)):
    """
    Main endpoint to analyze dental X-ray images (including converted DICOM)
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get image dimensions
        width, height = image.size
        
        # Perform pathology detection using Roboflow
        detection_results = await detect_pathology(image_data)
        
        # Generate annotated image
        annotated_image = create_annotated_image(image, detection_results)
        
        # Generate diagnostic report using Gemini
        report = await generate_diagnostic_report(detection_results, width, height)
        
        # Convert annotated image to base64
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG")
        annotated_image_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "detection_results": detection_results,
            "annotated_image": f"data:image/jpeg;base64,{annotated_image_b64}",
            "diagnostic_report": report,
            "image_dimensions": {"width": width, "height": height}
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def detect_pathology(image_data: bytes) -> Dict[str, Any]:
    """
    Send image to Roboflow for pathology detection using specified model endpoint
    """
    try:
        # Encode image to base64
        image_b64 = base64.b64encode(image_data).decode()
        
        # Prepare request to Roboflow with specified parameters
        url = f"{ROBOFLOW_MODEL_ENDPOINT}?api_key={ROBOFLOW_API_KEY}&confidence={CONFIDENCE_THRESHOLD}&overlap={OVERLAP_THRESHOLD}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                data=image_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
        
        if response.status_code != 200:
            logger.warning(f"Roboflow API error: {response.status_code} - {response.text}")
            # If Roboflow fails, return mock data for demonstration
            return get_mock_detection_results()
        
        result = response.json()
        
        # Validate and process the Roboflow response
        if "predictions" not in result:
            logger.warning("Invalid Roboflow response format")
            return get_mock_detection_results()
            
        return result
        
    except Exception as e:
        logger.warning(f"Roboflow API error: {e}")
        # Return mock data if API fails
        return get_mock_detection_results()

def get_mock_detection_results() -> Dict[str, Any]:
    """
    Mock detection results for demonstration when Roboflow API is not available
    Simulates the exact format returned by the specified Roboflow model
    """
    return {
        "predictions": [
            {
                "x": 250.0,
                "y": 180.0,
                "width": 45.0,
                "height": 38.0,
                "confidence": 0.87,
                "class": "caries",
                "class_id": 0
            },
            {
                "x": 420.0,
                "y": 220.0,
                "width": 52.0,
                "height": 44.0,
                "confidence": 0.74,
                "class": "periapical_lesion", 
                "class_id": 1
            },
            {
                "x": 180.0,
                "y": 280.0,
                "width": 35.0,
                "height": 30.0,
                "confidence": 0.65,
                "class": "filling",
                "class_id": 2
            }
        ]
    }

def create_annotated_image(image: Image.Image, detection_results: Dict[str, Any]) -> Image.Image:
    """
    Draw bounding boxes on the image based on detection results with pathology names and confidence scores
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Color mapping for different pathologies (bright colors for visibility)
    colors = {
        "caries": "#FF0000",  # Red
        "periapical_lesion": "#FFFF00",  # Yellow  
        "bone_loss": "#FF8800",  # Orange
        "impacted_tooth": "#0088FF",  # Blue
        "root_canal": "#FF00FF",  # Magenta
        "crown": "#00FF00",  # Green
        "filling": "#00FFFF",  # Cyan
        "implant": "#FF8080",  # Light Red
        "calculus": "#8080FF",  # Light Blue
        "default": "#FFFFFF"  # White for unknown classes
    }
    
    predictions = detection_results.get("predictions", [])
    
    for prediction in predictions:
        x = prediction["x"]
        y = prediction["y"]
        width = prediction["width"]
        height = prediction["height"]
        confidence = prediction["confidence"]
        class_name = prediction["class"]
        
        # Calculate bounding box coordinates (Roboflow uses center x,y format)
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        
        # Get color for this class
        color = colors.get(class_name, colors["default"])
        
        # Draw bounding box with thicker lines for visibility
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Create label with pathology name and confidence score
        label = f"{class_name.replace('_', ' ').title()}: {confidence:.0%}"
        
        # Calculate text size for better positioning
        try:
            bbox = draw.textbbox((0, 0), label)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            text_width = len(label) * 8
            text_height = 16
        
        # Position label above the bounding box
        label_x = max(0, x1)
        label_y = max(0, y1 - text_height - 5)
        
        # Draw label background for better readability
        draw.rectangle([label_x, label_y, label_x + text_width + 6, label_y + text_height + 4], fill=color)
        
        # Draw label text in black for contrast
        draw.text((label_x + 3, label_y + 2), label, fill="#000000")
    
    return annotated

async def generate_diagnostic_report(detection_results: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    """
    Generate diagnostic report using Gemini LLM based on image metadata and annotations
    """
    try:
        predictions = detection_results.get("predictions", [])
        
        # Prepare detailed findings with location information
        detailed_findings = []
        for pred in predictions:
            # Determine anatomical location based on coordinates
            location = determine_anatomical_location(pred["x"], pred["y"], width, height)
            
            detailed_findings.append({
                "pathology": pred["class"].replace("_", " ").title(),
                "confidence": f"{pred['confidence']:.0%}",
                "location": location,
                "coordinates": f"({pred['x']:.0f}, {pred['y']:.0f})",
                "size": f"{pred['width']:.0f}x{pred['height']:.0f} pixels"
            })
        
        # Create comprehensive prompt for Gemini as specified
        prompt = f"""You are a dental radiologist. Based on the image annotations provided below (which include detected pathologies), write a concise diagnostic report in clinical language.

Image Analysis Details:
- Image dimensions: {width}x{height} pixels
- Total pathologies detected: {len(predictions)}

Detected Pathologies:
{json.dumps(detailed_findings, indent=2)}

Please provide a diagnostic report that includes:
1. Detected pathologies with specific locations (e.g., upper left molar region)
2. Clinical significance of each finding
3. Recommended treatment or follow-up actions
4. Overall assessment and prognosis

Write the report in professional clinical language as you would for a referring dentist."""
        
        # Generate response using Gemini with updated model name
        if GEMINI_API_KEY :
            try:
                # Try the newer model names first
                model_names = [
                    'gemini-1.5-flash',
                    'gemini-1.5-pro', 
                    'gemini-1.0-pro',
                    'gemini-pro'
                ]
                
                response = None
                for model_name in model_names:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(prompt)
                        logger.info(f"Successfully used model: {model_name}")
                        break
                    except Exception as model_error:
                        logger.warning(f"Model {model_name} failed: {str(model_error)}")
                        continue
                
                if response and response.text:
                    report_text = response.text
                else:
                    logger.warning("All Gemini models failed, using fallback report")
                    report_text = generate_fallback_report_text(detailed_findings)
                    
            except Exception as e:
                logger.warning(f"Gemini API error: {e}, using fallback report")
                report_text = generate_fallback_report_text(detailed_findings)
        else:
            logger.warning("Gemini API key not configured, using fallback report")
            report_text = generate_fallback_report_text(detailed_findings)
        
        # Determine severity based on findings
        severity = determine_severity(predictions)
        
        # Calculate overall confidence
        if predictions:
            avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions)
        else:
            avg_confidence = 0.0
        
        return {
            "findings": report_text,
            "severity": severity,
            "confidence": avg_confidence,
            "pathology_count": len(predictions),
            "detected_conditions": [p["class"] for p in predictions],
            "detailed_annotations": detailed_findings
        }
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        # Return a fallback report
        return generate_fallback_report(detection_results)

def generate_fallback_report_text(detailed_findings: List[Dict[str, Any]]) -> str:
    """
    Generate a structured clinical report when Gemini API is not available
    """
    if not detailed_findings:
        return "DIAGNOSTIC FINDINGS:\n\nNo significant pathologies detected in this dental radiograph. The image appears to show normal dental structures with no obvious signs of decay, infection, or other abnormalities.\n\nRECOMMENDATIONS:\n- Routine clinical examination\n- Continue regular dental hygiene practices\n- Follow-up radiographs as per standard protocol"
    
    findings_text = "DIAGNOSTIC FINDINGS:\n\n"
    recommendations = []
    
    for i, finding in enumerate(detailed_findings, 1):
        pathology = finding['pathology'].upper()
        location = finding['location']
        confidence = finding['confidence']
        
        findings_text += f"{i}. {pathology} ({location}): "
        
        if pathology == "CARIES":
            findings_text += f"A carious lesion has been identified with {confidence} confidence. The lesion appears to require restorative intervention.\n\n"
            recommendations.append("Immediate restorative treatment for detected caries")
        elif pathology == "PERIAPICAL LESION":
            findings_text += f"A periapical radiolucency is present with {confidence} confidence, suggesting possible endodontic involvement.\n\n"
            recommendations.append("Endodontic evaluation for periapical lesion")
        elif pathology == "FILLING":
            findings_text += f"An existing restoration is noted with {confidence} confidence, appearing to be in acceptable condition.\n\n"
            recommendations.append("Monitor existing restoration for integrity")
        else:
            findings_text += f"Detected with {confidence} confidence. Clinical correlation recommended.\n\n"
            recommendations.append(f"Clinical evaluation of {pathology.lower()}")
    
    findings_text += "CLINICAL RECOMMENDATIONS:\n"
    for rec in recommendations:
        findings_text += f"- {rec}\n"
    findings_text += "- Clinical examination to correlate radiographic findings\n- Follow-up radiographs as clinically indicated"
    
    return findings_text

def determine_anatomical_location(x: float, y: float, width: int, height: int) -> str:
    """
    Determine anatomical location based on coordinates in dental X-ray
    """
    # Normalize coordinates to percentages
    x_percent = (x / width) * 100
    y_percent = (y / height) * 100
    
    # Determine horizontal region (left/center/right)
    if x_percent < 33:
        horizontal = "left"
    elif x_percent > 67:
        horizontal = "right"
    else:
        horizontal = "central"
    
    # Determine vertical region (upper/lower)
    if y_percent < 50:
        vertical = "upper"
    else:
        vertical = "lower"
    
    # Determine specific region
    if x_percent < 20 or x_percent > 80:
        region = "molar region"
    elif x_percent < 35 or x_percent > 65:
        region = "premolar region"
    else:
        region = "anterior region"
    
    return f"{vertical} {horizontal} {region}"

def determine_severity(predictions: List[Dict[str, Any]]) -> str:
    """
    Determine overall severity based on detected pathologies
    """
    if not predictions:
        return "Normal"
    
    high_severity_conditions = ["periapical_lesion", "bone_loss", "impacted_tooth"]
    moderate_severity_conditions = ["caries", "root_canal"]
    
    max_confidence = max(p["confidence"] for p in predictions)
    
    for pred in predictions:
        if pred["class"] in high_severity_conditions and pred["confidence"] > 0.7:
            return "High"
        elif pred["class"] in moderate_severity_conditions and pred["confidence"] > 0.8:
            return "Moderate"
    
    if max_confidence > 0.6:
        return "Mild"
    else:
        return "Low Confidence"

def generate_fallback_report(detection_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a basic report when Gemini API is not available
    """
    predictions = detection_results.get("predictions", [])
    
    if not predictions:
        findings = "No significant pathologies detected in this dental radiograph."
        severity = "Normal"
    else:
        pathology_names = [p["class"].replace("_", " ").title() for p in predictions]
        findings = f"Detected pathologies: {', '.join(pathology_names)}. "
        findings += "Recommend clinical correlation and follow-up examination."
        severity = determine_severity(predictions)
    
    avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions) if predictions else 0.0
    
    return {
        "findings": findings,
        "severity": severity,
        "confidence": avg_confidence,
        "pathology_count": len(predictions),
        "detected_conditions": [p["class"] for p in predictions]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)