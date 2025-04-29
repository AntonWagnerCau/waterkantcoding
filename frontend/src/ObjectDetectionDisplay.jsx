import React, { useState, useEffect } from 'react';
import ThreeDVisualization from './ThreeDVisualization';

// Assuming helper functions or context might be used for consistent styling later
// For now, we rely on CSS classes defined in index.css

const ObjectDetectionDisplay = ({ objectDetection, odometry }) => {
  const [enlargedImage, setEnlargedImage] = useState(null);

  const formatCameraName = (name) => {
    const mapping = {
      'frontleft_fisheye_image': 'Front Left',
      'frontright_fisheye_image': 'Front Right',
      'left_fisheye_image': 'Left',
      'right_fisheye_image': 'Right',
      'back_fisheye_image': 'Back'
    };
    return mapping[name] || name;
  };

  const formatConfidence = (score) => {
    if (typeof score === 'number') {
      return `${(score * 100).toFixed(0)}%`;
    }
    return score;
  };

  const handleImageClick = (base64String) => {
    setEnlargedImage(`data:image/jpeg;base64,${base64String}`);
  };

  const closeEnlargedImage = () => {
    setEnlargedImage(null);
  };

  // --- Loading State --- 
  if (!objectDetection || !objectDetection.status || objectDetection.status === 'pending') {
    return (
      <div className="panel object-detection-panel placeholder-panel"> {/* Added specific class */} 
        <div className="panel-header">
            <h2>OBJECT DETECTION</h2>
        </div>
        <div className="panel-body">
          <p className="status-text waiting">Awaiting object detection data stream...</p>
        </div>
      </div>
    );
  }

  // --- Error State --- 
  if (objectDetection.status === 'error') {
    return (
      <div className="panel object-detection-panel error-panel"> {/* Added specific class */} 
        <div className="panel-header">
          <h2>OBJECT DETECTION ERROR</h2>
        </div>
        <div className="panel-body">
          <p className="status-text error">Error: {objectDetection.error || 'Unknown error'}</p>
        </div>
      </div>
    );
  }

  // --- Success State --- 
  const base64Images = objectDetection.base64_images || {};
  const imageKeys = Object.keys(base64Images);

  // Define the desired display order
  const desiredOrder = [
    'left_fisheye_image',
    'frontright_fisheye_image',
    'frontleft_fisheye_image',
    'right_fisheye_image',
    'back_fisheye_image'
  ];

  // Sort the image keys based on the desired order
  const sortedImageKeys = imageKeys.sort((a, b) => {
    const indexA = desiredOrder.indexOf(a);
    const indexB = desiredOrder.indexOf(b);

    // Handle cases where a key might not be in the desired order (place them at the end)
    if (indexA === -1 && indexB === -1) return 0; // Keep original relative order if both are unknown
    if (indexA === -1) return 1;  // Place unknown keys after known keys
    if (indexB === -1) return -1; // Place known keys before unknown keys

    return indexA - indexB;
  });

  return (
    <div className="panel object-detection-panel"> {/* Added specific class */} 
      <div className="panel-header">
        <h2>OBJECT DETECTION</h2>
        <div className="panel-controls"> {/* Optional: Add controls later if needed */} 
           <span>Detected: {objectDetection.object_count ?? 0}</span>
        </div>
      </div>
      <div className="panel-body">
        {/* 3D Visualization */}
        <div className="three-d-visualization-container">
          <h3>3D Visualization</h3>
          <ThreeDVisualization objectDetection={objectDetection} odometry={odometry} />
        </div>
        {/* Image Grid - Needs corresponding CSS in index.css */}
        <div className="camera-image-grid">
          {sortedImageKeys.length > 0 ? sortedImageKeys.map((camera) => (
            <div key={camera} className="camera-view-item"> {/* Custom class for styling */} 
              <h3 className="camera-view-title">{formatCameraName(camera)}</h3>
              <div className="camera-image-container"> {/* Custom class */} 
                <img
                  src={`data:image/jpeg;base64,${base64Images[camera]}`}
                  alt={`View from ${formatCameraName(camera)}`}
                  className="camera-image" // Custom class
                  onClick={() => handleImageClick(base64Images[camera])}
                  onError={(e) => {
                     e.target.onerror = null; 
                     e.target.alt = `Error loading ${formatCameraName(camera)}`;
                     e.target.style.display = 'none'; // Hide broken image potentially
                     console.error(`[ObjectDetectionDisplay] Error loading image for ${camera}.`); 
                  }}
                />
                 {/* Optional: Add overlays like the old camera feed? */}
              </div>
            </div>
          )) : (
            <p className="status-text info col-span-full">No camera images available in this update.</p> // Use theme class
          )}
        </div>

        {/* Detected Objects List - Needs corresponding CSS */}
        {objectDetection.objects && objectDetection.objects.length > 0 && (
          <div className="object-list-container"> {/* Custom class */} 
            <h3 className="object-list-title">Detected Objects</h3>
            <ul className="object-list"> {/* Custom class */} 
              {objectDetection.objects.map((obj, index) => (
                <li key={index} className="object-list-item"> {/* Custom class */} 
                  {obj.label} ({formatConfidence(obj.score)}) 
                  <span className='object-source'> | Source: {formatCameraName(obj.source_camera || obj.source || 'unknown')}</span>
                  {/* Display 3D Coordinates if available */}
                  {obj.position && (
                     <span className='object-position'>
                       | Pos (xyz): ({obj.position.x?.toFixed(2) ?? 'N/A'}, {obj.position.y?.toFixed(2) ?? 'N/A'}, {obj.position.z?.toFixed(2) ?? 'N/A'}) m
                     </span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Modal for enlarged image - Check if styling needs adjustments */} 
      {enlargedImage && (
        <div
          className="image-modal-overlay" // Use themed class
          onClick={closeEnlargedImage}
        >
          <div className="image-modal-content" onClick={(e) => e.stopPropagation()}> 
            <img src={enlargedImage} alt="Enlarged View" className="image-modal-image" />
          </div>
        </div>
      )}
    </div>
  );
};

export default ObjectDetectionDisplay; 