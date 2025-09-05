import { Download, Eye, AlertCircle, CheckCircle2, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import type { PredictionResult } from "@/types";

interface ResultsSectionProps {
  result: PredictionResult | null;
  uploadedImage: string | null;
  isLoading: boolean;
}

export const ResultsSection = ({ result, uploadedImage, isLoading }: ResultsSectionProps) => {
  if (!uploadedImage && !result) return null;

  const generatePDFReport = () => {
    // Mock PDF generation - in real app, this would call backend API
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,AI Eye Scan Report\n\nDisease: ' + result?.disease + '\nConfidence: ' + (result?.confidence ? Math.round(result.confidence * 100) : 0) + '%\nDate: ' + new Date().toLocaleDateString());
    element.setAttribute('download', 'eye-scan-report.txt');
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const getResultColor = (disease: string) => {
    if (disease.toLowerCase() === 'normal') return 'text-neon-green';
    if (disease.toLowerCase().includes('diabetic')) return 'text-neon-orange';
    return 'text-neon-pink';
  };

  const getResultIcon = (disease: string) => {
    if (disease.toLowerCase() === 'normal') return CheckCircle2;
    return AlertCircle;
  };

  return (
    <section className="py-20 px-6 bg-muted/20">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-neon-purple to-neon-pink bg-clip-text text-transparent">
            Analysis Results
          </h2>
          <p className="text-muted-foreground text-lg">
            AI-powered eye disease detection results
          </p>
        </div>

        {isLoading && (
          <Card className="border-neon-cyan/30 bg-card/80 backdrop-blur-sm">
            <CardContent className="p-8 text-center">
              <div className="animate-pulse space-y-4">
                <div className="w-16 h-16 bg-neon-cyan/20 rounded-full mx-auto"></div>
                <div className="h-4 bg-muted rounded w-3/4 mx-auto"></div>
                <div className="h-3 bg-muted rounded w-1/2 mx-auto"></div>
              </div>
            </CardContent>
          </Card>
        )}

        {result && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Main Result Card */}
            <Card className="border-neon-cyan/30 bg-card/80 backdrop-blur-sm animate-fade-in-up">
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  {(() => {
                    const Icon = getResultIcon(result.disease);
                    return <Icon className={`w-6 h-6 ${getResultColor(result.disease)}`} />;
                  })()}
                  Detection Result
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className={`text-2xl font-bold mb-2 ${getResultColor(result.disease)}`}>
                    {result.disease}
                  </h3>
                  <Badge 
                    variant="secondary"
                    className="bg-neon-cyan/10 text-neon-cyan border-neon-cyan/30"
                  >
                    AI Confidence: {Math.round(result.confidence * 100)}%
                  </Badge>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Confidence Score</span>
                    <span>{Math.round(result.confidence * 100)}%</span>
                  </div>
                  <Progress 
                    value={result.confidence * 100} 
                    className="h-3"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4 pt-4">
                  <Button
                    variant="outline"
                    className="border-neon-purple/50 hover:border-neon-purple hover:bg-neon-purple/10"
                  >
                    <Eye className="w-4 h-4 mr-2" />
                    View Heatmap
                  </Button>
                  <Button
                    onClick={generatePDFReport}
                    className="bg-gradient-to-r from-neon-cyan to-neon-purple hover:shadow-neon"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download Report
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Analysis Details Card */}
            <Card className="border-neon-purple/30 bg-card/80 backdrop-blur-sm animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <TrendingUp className="w-6 h-6 text-neon-purple" />
                  Analysis Details
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                    <span className="text-sm">Model Used</span>
                    <Badge variant="secondary">EfficientNetB0</Badge>
                  </div>
                  
                  <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                    <span className="text-sm">Processing Time</span>
                    <span className="text-sm text-neon-cyan">2.3s</span>
                  </div>
                  
                  <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                    <span className="text-sm">Image Quality</span>
                    <span className="text-sm text-neon-green">Excellent</span>
                  </div>
                  
                  <div className="flex justify-between items-center p-3 bg-muted/50 rounded-lg">
                    <span className="text-sm">Analysis Date</span>
                    <span className="text-sm">{new Date().toLocaleDateString()}</span>
                  </div>
                </div>

                <div className="pt-4 border-t border-border">
                  <h4 className="text-sm font-semibold mb-2">Recommendation</h4>
                  <p className="text-sm text-muted-foreground">
                    {result.disease.toLowerCase() === 'normal' 
                      ? "Your retinal scan appears normal. Continue regular eye exams as recommended by your eye care professional."
                      : "Please consult with an eye care professional for proper evaluation and treatment options. This AI analysis is for informational purposes only."
                    }
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </section>
  );
};