package amten.ml.examples;

import javax.print.DocFlavor.URL;

import amten.ml.NNParams;
import amten.ml.matrix.Matrix;
import amten.ml.matrix.MatrixUtils;

import java.io.*;
import java.util.Arrays;
import java.util.Collections;


/**
 * Examples of using NeuralNetwork for classification.
 *
 * @author Johannes Amtén
 * and Cheng-Han Lin
 */
public class NNClassificationExample {

    /**
     * Performs classification of Handwritten digits,
     * using a subset (1000 rows) from the Kaggle Digits competition.
     * <br></br>
     * Uses file /example_data/Kaggle_Digits_1000.csv
     *
     * @see <a href="http://www.kaggle.com/c/digit-recognizer">http://www.kaggle.com/c/digit-recognizer</a></a>
     */
	static String outputText="";
	
	
    //public static void runKaggleDigitsClassification(boolean useConvolution) throws Exception {
	public static void runKaggleDigitsClassification(boolean useConvolution, String filename_train
			, String filename_test
			) throws Exception {
	if (useConvolution) {
            System.out.println("Running classification on Kaggle Digits dataset, with convolution...\n");
        } else {
            System.out.println("Running classification on Kaggle Digits dataset...\n");
        }
        // Read data from CSV-file
        int headerRows = 1;
        char separator = ',';
        
        //Matrix data = MatrixUtils.readCSV("train200.csv", separator, headerRows);
        Matrix data = MatrixUtils.readCSV(filename_train, separator, headerRows);

        // Split data into training set and crossvalidation set.
        float crossValidationPercent = 0; //33;
        Matrix[] split = MatrixUtils.split(data, crossValidationPercent, 0);
        Matrix dataTrain = split[0];
        //Matrix dataCV = split[1];
        
        //Matrix dataCV = MatrixUtils.readCSV("test.csv", separator, headerRows);
        Matrix dataCV = MatrixUtils.readCSV(filename_test, separator, headerRows);
        
        
        // First column contains the classification label. The rest are the indata.
        Matrix xTrain = dataTrain.getColumns(1, -1);
        Matrix yTrain = dataTrain.getColumns(0, 0);
        Matrix xCV = dataCV.getColumns(1, -1);
        Matrix yCV = dataCV.getColumns(0, 0);

        
        NNParams params = new NNParams();
        params.numClasses = 10; // 10 digits to classify
        params.hiddenLayerParams = useConvolution ? new NNParams.NNLayerParams[]{ new NNParams.NNLayerParams(20, 5, 5, 2, 2) , new NNParams.NNLayerParams(100, 5, 5, 2, 2) } :
        											new NNParams.NNLayerParams[] { new NNParams.NNLayerParams(100) };
        params.maxIterations = useConvolution ? 10 : 200;
        params.learningRate = useConvolution ? 1E-2 : 0;
        /*Hidden layers are specified as comma-separated lists.
        e.g. "100,100" for two layers with 100 units each.
        For convolutional layers: <num feature maps>-<patch-width>-<patch-height>-<pool-width>-<pool-height> 
        e.g. "20-5-5-2-2,100-5-5-2-2" for two convolutional layers, both with patch size 5x5 and pool size 2x2, each with 20 and 100 feature maps respectively.
        */
        long startTime = System.currentTimeMillis();
        amten.ml.NeuralNetwork nn = new amten.ml.NeuralNetwork(params);
        nn.train(xTrain, yTrain);
        System.out.println("\nTraining time: " + String.format("%.3g", (System.currentTimeMillis() - startTime) / 1000.0) + "s");

/*        
        try
        {
           FileOutputStream fileOut =
           new FileOutputStream("TrainedConvolutionalNeuralNetwork.ser");
           //new FileOutputStream("TrainedNeuralNetwork.ser");
           ObjectOutputStream out = new ObjectOutputStream(fileOut);
           out.writeObject(nn);
           out.close();
           fileOut.close();
           System.out.printf("Serialized data is saved in TrainedConvolutionalNeuralNetwork.ser\n");
           //System.out.printf("Serialized data is saved in TrainedNeuralNetwork.ser\n");
        }catch(IOException i)
        {
            i.printStackTrace();
        }
 */    
 /*       
        amten.ml.NeuralNetwork nn = null;
        try
        {
           FileInputStream fileIn = new FileInputStream("TrainedNeuralNetwork.ser");
           //FileInputStream fileIn = new FileInputStream("TrainedConvolutionalNeuralNetwork.ser");
           ObjectInputStream in = new ObjectInputStream(fileIn);
           nn = (amten.ml.NeuralNetwork) in.readObject();
           in.close();
           fileIn.close();
        }catch(IOException i)
        {
           i.printStackTrace();
           return;
        }catch(ClassNotFoundException c)
        {
           System.out.println("amten.ml.NeuralNetwork class not found\n");
           c.printStackTrace();
           return;
        }
*/
        
        int[] predictedClasses = nn.getPredictedClasses(xTrain);
        int correct = 0;
        for (int i = 0; i < predictedClasses.length; i++) {
            if (predictedClasses[i] == yTrain.get(i, 0)) {
                correct++;
            }
        }
        
        System.out.println("Training set accuracy: " + String.format("%.3g", (double) correct/predictedClasses.length*100) + "%");

        
        predictedClasses = nn.getPredictedClasses(xCV);
        correct = 0;
        for (int i = 0; i < predictedClasses.length; i++) {
        
        	outputText= outputText+predictedClasses[i]+"\r\n";
            if (predictedClasses[i] == yCV.get(i, 0)) {	
            	System.out.println("Predicted Class is: "+predictedClasses[i]); //added
                correct++;
            }
        }
        System.out.println(//"Crossvalidation "
        		"Test set accuracy: " + String.format("%.3g", (double) correct/predictedClasses.length*100) + "%");
        
        //if (useConvolution){
        PrintWriter out = new PrintWriter("../NeuralNetworkResult.txt");
        out.println(outputText);
        out.close();
        //}
        
        
    }

  
    public static void main(String[] args) throws Exception {
    	

    	System.out.println("arg[0] "+args[0]+"; args[1]"+ args[1]);
    	runKaggleDigitsClassification(false,args[0], args[1]);
    	System.out.println();
        runKaggleDigitsClassification(true,args[0], args[1]);

    }
}