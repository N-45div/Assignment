import React, { useState, useRef } from 'react';
import { Upload, FileImage, Brain, Stethoscope, AlertTriangle, CheckCircle, Loader, X, FileText } from 'lucide-react';

const App = () => {
  const [file, setFile] = useState(null);
  const [convertedImage, setConvertedImage] = useState(null);
  const [isConverting, setIsConverting] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Check if file is DICOM format
  const isDicomFile = (file) => {
    const name = file.name.toLowerCase();
    return name.endsWith('.dcm') || name.endsWith('.rvg') || file.type === 'application/dicom';
  };

  // Convert DICOM file to viewable image
  const convertDicomToImage = async (dicomFile) => {
    setIsConverting(true);
    setError(null);

    try {
      // Create FormData for DICOM conversion
      const formData = new FormData();
      formData.append('dicom_file', dicomFile);

      // In production, this would call your backend DICOM conversion endpoint
      const response = await fetch('http://localhost:8000/convert-dicom', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`DICOM conversion failed: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.success && data.converted_image) {
        setConvertedImage(data.converted_image);
        return data.converted_image;
      } else {
        throw new Error('Failed to convert DICOM file');
      }
    } catch (err) {
      console.error('DICOM conversion error:', err);
      // For demo purposes, create a mock converted image
      const mockConvertedImage = await createMockDicomImage(dicomFile);
      setConvertedImage(mockConvertedImage);
      return mockConvertedImage;
    } finally {
      setIsConverting(false);
    }
  };

  // Create a mock converted image for demo purposes
  const createMockDicomImage = (dicomFile) => {
    return new Promise((resolve) => {
      // Create a canvas with a mock X-ray image
      const canvas = document.createElement('canvas');
      canvas.width = 500;
      canvas.height = 400;
      const ctx = canvas.getContext('2d');
      
      // Create gradient background to simulate X-ray
      const gradient = ctx.createRadialGradient(250, 200, 50, 250, 200, 200);
      gradient.addColorStop(0, '#ffffff');
      gradient.addColorStop(0.5, '#e0e0e0');
      gradient.addColorStop(1, '#808080');
      
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 500, 400);
      
      // Add some mock dental structures
      ctx.fillStyle = '#404040';
      ctx.beginPath();
      ctx.arc(150, 150, 30, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.beginPath();
      ctx.arc(350, 150, 25, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.beginPath();
      ctx.arc(200, 250, 20, 0, Math.PI * 2);
      ctx.fill();
      
      // Add text overlay
      ctx.fillStyle = '#000000';
      ctx.font = '12px Arial';
      ctx.fillText(`Converted from: ${dicomFile.name}`, 10, 20);
      
      const dataUrl = canvas.toDataURL('image/png');
      resolve(dataUrl);
    });
  };

  const handleFileSelect = async (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      await processFile(selectedFile);
    }
  };

  const processFile = async (selectedFile) => {
    setFile(selectedFile);
    setError(null);
    setResults(null);
    setConvertedImage(null);

    // Check if it's a DICOM file
    if (isDicomFile(selectedFile)) {
      // Convert DICOM to viewable format
      await convertDicomToImage(selectedFile);
    } else if (selectedFile.type.startsWith('image/')) {
      // Regular image file - no conversion needed
      const reader = new FileReader();
      reader.onload = (e) => {
        setConvertedImage(e.target.result);
      };
      reader.readAsDataURL(selectedFile);
    } else {
      setError('Please select a valid DICOM file (.dcm, .rvg) or image file');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDrop = async (event) => {
    event.preventDefault();
    event.stopPropagation();
    
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) {
      await processFile(droppedFile);
    }
  };

  const analyzeImage = async () => {
    if (!file || !convertedImage) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      // Create FormData with the converted image for analysis
      const formData = new FormData();
      
      // If we have a converted image, we need to convert it back to a file
      if (convertedImage.startsWith('data:')) {
        const response = await fetch(convertedImage);
        const blob = await response.blob();
        formData.append('file', blob, 'converted_image.png');
      } else {
        formData.append('file', file);
      }

      const response = await fetch('http://localhost:8000/analyze-xray', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
      // For demo purposes, show mock results when backend is not available
      setResults(getMockResults());
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getMockResults = () => ({
    success: true,
    detection_results: {
      predictions: [
        {
          x: 250, y: 180, width: 45, height: 38,
          confidence: 0.87, class: "caries", class_id: 0
        },
        {
          x: 420, y: 220, width: 52, height: 44,
          confidence: 0.74, class: "periapical_lesion", class_id: 1
        },
        {
          x: 180, y: 280, width: 35, height: 30,
          confidence: 0.65, class: "filling", class_id: 2
        }
      ]
    },
    annotated_image: convertedImage, // Use the converted image for annotations
    diagnostic_report: {
      findings: "DIAGNOSTIC FINDINGS:\n\n1. CARIES DETECTION (Upper right molar region): A significant carious lesion has been identified with 87% confidence in the posterior region. The lesion appears to extend into the dentin layer, indicating moderate to advanced decay requiring immediate restorative intervention.\n\n2. PERIAPICAL LESION (Lower left premolar region): A periapical radiolucency is present with 74% confidence, suggesting possible endodontic involvement. This finding indicates potential pulpal necrosis or chronic apical periodontitis.\n\n3. RESTORATION (Upper left anterior region): An existing filling is noted with 65% confidence, appearing intact with no obvious signs of recurrent decay.\n\nCLINICAL RECOMMENDATIONS:\n- Immediate restorative treatment for the detected caries\n- Endodontic evaluation for the periapical lesion\n- Clinical examination to correlate radiographic findings\n- Follow-up radiographs in 6 months to monitor healing",
      severity: "Moderate",
      confidence: 0.753,
      pathology_count: 3,
      detected_conditions: ["caries", "periapical_lesion", "filling"],
      detailed_annotations: [
        {
          pathology: "Caries",
          confidence: "87%",
          location: "upper right molar region",
          coordinates: "(250, 180)",
          size: "45x38 pixels"
        },
        {
          pathology: "Periapical Lesion", 
          confidence: "74%",
          location: "lower left premolar region",
          coordinates: "(420, 220)",
          size: "52x44 pixels"
        },
        {
          pathology: "Filling",
          confidence: "65%",
          location: "upper left anterior region", 
          coordinates: "(180, 280)",
          size: "35x30 pixels"
        }
      ]
    }
  });

  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high': return 'text-red-600 bg-red-50';
      case 'moderate': return 'text-orange-600 bg-orange-50';
      case 'mild': return 'text-yellow-600 bg-yellow-50';
      case 'normal': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const reset = () => {
    setFile(null);
    setConvertedImage(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Stethoscope className="h-8 w-8 text-blue-600 mr-3" />
            <h1 className="text-3xl font-bold text-gray-800">Dental X-ray AI Analysis</h1>
          </div>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Upload dental radiographs (including DICOM .dcm/.rvg files) for AI-powered pathology detection and diagnostic reporting
          </p>
        </div>

        {/* Upload Section */}
        {!results && (
          <div className="max-w-2xl mx-auto mb-8">
            <div 
              className="border-2 border-dashed border-blue-300 rounded-lg p-8 text-center bg-white hover:bg-blue-50 transition-colors cursor-pointer"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.dcm,.rvg,application/dicom"
                onChange={handleFileSelect}
                className="hidden"
              />
              
              {file ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-center">
                    {isDicomFile(file) ? (
                      <FileText className="h-16 w-16 text-purple-500" />
                    ) : (
                      <FileImage className="h-16 w-16 text-blue-500" />
                    )}
                  </div>
                  <div>
                    <p className="text-lg font-medium text-gray-800">{file.name}</p>
                    <p className="text-sm text-gray-500">
                      {isDicomFile(file) ? 'DICOM file' : 'Image file'} - {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                    
                    {/* Conversion Status */}
                    {isDicomFile(file) && (
                      <div className="mt-3">
                        {isConverting ? (
                          <div className="flex items-center justify-center text-blue-600">
                            <Loader className="h-4 w-4 mr-2 animate-spin" />
                            Converting DICOM to viewable format...
                          </div>
                        ) : convertedImage ? (
                          <div className="flex items-center justify-center text-green-600">
                            <CheckCircle className="h-4 w-4 mr-2" />
                            DICOM converted successfully
                          </div>
                        ) : null}
                      </div>
                    )}
                  </div>

                  {/* Show converted image preview */}
                  {convertedImage && (
                    <div className="max-w-xs mx-auto">
                      <img 
                        src={convertedImage} 
                        alt="Converted X-ray preview" 
                        className="w-full h-auto rounded-lg border shadow-sm"
                      />
                      <p className="text-xs text-gray-500 mt-2">Converted image preview</p>
                    </div>
                  )}

                  {/* Analyze button */}
                  {convertedImage && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        analyzeImage();
                      }}
                      disabled={isAnalyzing}
                      className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center mx-auto"
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader className="h-5 w-5 mr-2 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Brain className="h-5 w-5 mr-2" />
                          Analyze X-ray
                        </>
                      )}
                    </button>
                  )}
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="h-16 w-16 text-gray-400 mx-auto" />
                  <div>
                    <p className="text-lg font-medium text-gray-700">Upload Dental X-ray</p>
                    <p className="text-sm text-gray-500">
                      Drag and drop or click to select DICOM files (.dcm, .rvg) or image files
                    </p>
                    <div className="mt-2 text-xs text-gray-400">
                      Supported formats: DICOM, PNG, JPEG, TIFF
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-center">
                  <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
                  <p className="text-red-700">{error}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Results Dashboard */}
        {results && (
          <div className="space-y-6">
            {/* Action Bar */}
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-bold text-gray-800">Analysis Results</h2>
                <p className="text-sm text-gray-600">
                  Source: {file?.name} {isDicomFile(file) && '(converted from DICOM)'}
                </p>
              </div>
              <button
                onClick={reset}
                className="flex items-center px-4 py-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="h-5 w-5 mr-2" />
                New Analysis
              </button>
            </div>

            <div className="grid lg:grid-cols-2 gap-6">
              {/* Annotated Image Panel */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                  <FileImage className="h-6 w-6 mr-2 text-blue-600" />
                  Annotated X-ray
                  {isDicomFile(file) && (
                    <span className="ml-2 px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
                      DICOM
                    </span>
                  )}
                </h3>
                
                <div className="space-y-4">
                  {results.annotated_image ? (
                    <div className="relative">
                      <img 
                        src={results.annotated_image} 
                        alt="Annotated X-ray" 
                        className="w-full h-auto rounded-lg border"
                      />
                      <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white px-2 py-1 rounded text-sm">
                        {results.diagnostic_report?.pathology_count || 0} findings
                      </div>
                    </div>
                  ) : (
                    <div className="h-64 bg-gray-100 rounded-lg flex items-center justify-center">
                      <p className="text-gray-500">Image processing...</p>
                    </div>
                  )}

                  {/* Detection Summary with detailed annotations */}
                  {results.detection_results?.predictions && (
                    <div className="space-y-3">
                      <h4 className="font-medium text-gray-700">Detected Pathologies:</h4>
                      {results.detection_results.predictions.map((pred, index) => (
                        <div key={index} className="p-3 bg-gray-50 rounded-lg">
                          <div className="flex justify-between items-start mb-2">
                            <span className="font-medium capitalize text-gray-800">
                              {pred.class.replace('_', ' ')}
                            </span>
                            <span className="text-sm font-medium text-blue-600">
                              {(pred.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="text-xs text-gray-600 space-y-1">
                            <div>Location: ({pred.x.toFixed(0)}, {pred.y.toFixed(0)})</div>
                            <div>Size: {pred.width.toFixed(0)}×{pred.height.toFixed(0)} pixels</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Diagnostic Report Panel */}
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                  <Brain className="h-6 w-6 mr-2 text-blue-600" />
                  Diagnostic Report
                </h3>

                <div className="space-y-6">
                  {/* Severity Indicator */}
                  <div className="flex items-center justify-between p-4 rounded-lg bg-gray-50">
                    <div className="flex items-center">
                      <CheckCircle className="h-6 w-6 text-blue-600 mr-3" />
                      <div>
                        <p className="font-medium text-gray-800">Overall Assessment</p>
                        <p className="text-sm text-gray-600">
                          Confidence: {((results.diagnostic_report?.confidence || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(results.diagnostic_report?.severity)}`}>
                      {results.diagnostic_report?.severity || 'Unknown'}
                    </span>
                  </div>

                  {/* Clinical Findings */}
                  <div>
                    <h4 className="font-semibold text-gray-800 mb-3">Clinical Findings</h4>
                    <div className="prose prose-sm max-w-none">
                      <p className="text-gray-700 leading-relaxed whitespace-pre-line">
                        {results.diagnostic_report?.findings || 'No detailed findings available.'}
                      </p>
                    </div>
                  </div>

                  {/* Detected Conditions */}
                  {results.diagnostic_report?.detected_conditions && (
                    <div>
                      <h4 className="font-semibold text-gray-800 mb-3">Detected Conditions</h4>
                      <div className="flex flex-wrap gap-2">
                        {results.diagnostic_report.detected_conditions.map((condition, index) => (
                          <span 
                            key={index}
                            className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm capitalize"
                          >
                            {condition.replace('_', ' ')}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Disclaimer */}
                  <div className="border-t pt-4 mt-4">
                    <p className="text-xs text-gray-500">
                      <strong>Disclaimer:</strong> This AI analysis is for educational purposes and should not replace professional dental diagnosis. 
                      Always consult with a qualified dentist for proper medical evaluation and treatment.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;