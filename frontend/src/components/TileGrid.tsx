import React, { useState, ChangeEvent, useRef } from 'react';
import WebcamTile from './WebcamTile';
import ImageTile from './ImageTile';
import { ImageTileData } from '../types';

const TileGrid: React.FC = () => {
  const [images, setImages] = useState<ImageTileData[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleCapture = (imageDataUrl: string) => {
    const newImage: ImageTileData = {
      id: `img-${Date.now()}`,
      src: imageDataUrl,
    };
    setImages(prevImages => [newImage, ...prevImages]); // Add new image to the beginning
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const imageDataUrl = e.target?.result as string;
        if (imageDataUrl) {
          const newImage: ImageTileData = {
            id: `uploaded-${file.name}-${Date.now()}`,
            src: imageDataUrl,
          };
          setImages(prevImages => [newImage, ...prevImages]); // Add new image to the beginning
        }
      };
      reader.readAsDataURL(file);
      // Reset file input value to allow uploading the same file again
      if(fileInputRef.current) {
          fileInputRef.current.value = "";
      }
    }
  };

  const triggerFileUpload = () => {
      fileInputRef.current?.click();
  }

  return (
    <div>
      <div style={{ marginBottom: '10px' }}>
        <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }} // Hide the default input
            onChange={handleFileChange}
            accept="image/*" // Accept only image files
        />
        <button onClick={triggerFileUpload}>Upload Image</button>
      </div>
      <div className="tile-grid" style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
        <WebcamTile onCapture={handleCapture} />
        {images.map((image) => (
          <ImageTile key={image.id} image={image} />
        ))}
      </div>
    </div>
  );
};

export default TileGrid; 