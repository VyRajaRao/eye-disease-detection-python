import { useState } from "react";
import { HeroSection } from "@/components/HeroSection";
import { UploadSection } from "@/components/UploadSection";
import { ResultsSection } from "@/components/ResultsSection";
import { AboutSection } from "@/components/AboutSection";
import { DisclaimerSection } from "@/components/DisclaimerSection";
import { Footer } from "@/components/Footer";
import { ChatBot } from "@/components/ChatBot";
import type { PredictionResult } from "@/types";

const Index = () => {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleImageUpload = async (imageFile: File, imageUrl: string) => {
    setUploadedImage(imageUrl);
    setIsLoading(true);
    
    // Simulate API call for now
    setTimeout(() => {
      const mockResults = [
        { disease: "Diabetic Retinopathy", confidence: 0.89 },
        { disease: "Glaucoma", confidence: 0.76 },
        { disease: "Cataract", confidence: 0.92 },
        { disease: "Normal", confidence: 0.95 }
      ];
      
      const result = mockResults[Math.floor(Math.random() * mockResults.length)];
      setPredictionResult(result);
      setIsLoading(false);
    }, 3000);
  };

  const resetAnalysis = () => {
    setUploadedImage(null);
    setPredictionResult(null);
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="relative">
        <HeroSection />
        <UploadSection 
          onImageUpload={handleImageUpload}
          uploadedImage={uploadedImage}
          isLoading={isLoading}
          onReset={resetAnalysis}
        />
        <ResultsSection 
          result={predictionResult}
          uploadedImage={uploadedImage}
          isLoading={isLoading}
        />
        <AboutSection />
        <DisclaimerSection />
        <Footer />
        <ChatBot />
      </div>
    </div>
  );
};

export default Index;