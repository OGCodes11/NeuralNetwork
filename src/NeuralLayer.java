import java.io.*;

public class NeuralLayer {
    private RELUNeuron[] neurons;

    public NeuralLayer(int numInputs, int numNeurons) {
        neurons = new RELUNeuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new RELUNeuron(numInputs);
        }
    }

    public double[] output(double[] inputs) {
        double[] results = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            results[i] = neurons[i].output(inputs);
        }
        return results;
    }

    public void write(DataOutputStream out) throws IOException {
        out.writeInt(neurons.length);
        for (RELUNeuron neuron : neurons) {
            neuron.write(out);
        }
    }

    public void read(DataInputStream in) throws IOException {
        int length = in.readInt();
        neurons = new RELUNeuron[length];
        for (int i = 0; i < length; i++) {

            neurons[i] = new RELUNeuron(0);
            neurons[i].read(in);
        }
    }
}
