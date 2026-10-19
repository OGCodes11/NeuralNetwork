import java.io.*;

public class NeuralNetwork {
    private NeuralLayer firstLayer;
    private NeuralLayer secondLayer;
    private LogisticNeuron outputNeuron;

    public NeuralNetwork(int numInputs) {
        firstLayer = new NeuralLayer(numInputs, 100);
        secondLayer = new NeuralLayer(100, 10);
        outputNeuron = new LogisticNeuron(10);
    }

    public double output(double[] inputs) {
        double[] firstOut = firstLayer.output(inputs);
        double[] secondOut = secondLayer.output(firstOut);
        return outputNeuron.output(secondOut);
    }

    public void train(double[] inputs, double target) {
        double[] firstOut = firstLayer.output(inputs);
        double[] secondOut = secondLayer.output(firstOut);
        outputNeuron.train(secondOut, target);
    }

    public void write(DataOutputStream out) throws IOException {
        firstLayer.write(out);
        secondLayer.write(out);
        outputNeuron.write(out);
    }

    public void read(DataInputStream in) throws IOException {
        firstLayer.read(in);
        secondLayer.read(in);
        outputNeuron.read(in);
    }
}