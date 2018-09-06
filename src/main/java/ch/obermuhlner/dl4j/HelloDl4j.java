package ch.obermuhlner.dl4j;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import static java.lang.Math.toIntExact;

public class HelloDl4j {
    private final Random random = new Random(12345);

    public Path modelDir = Paths.get(".");
    public Path dataDir = Paths.get(".");
    public Path testDir = null;
    public Path trainDir = null;
    public String modelFileName = "model.dl4j";

    public int width = 100;
    public int height = 100;
    public int channels = 3;
    public int epochs = 10;
    public int batchSize = 20;
    public double testFraction = 0.2;
    public List<String> labels = new ArrayList<>();

    public void test() {

    }

    public void create() throws IOException {
        init();
        int numLabels = labels.size();

        MultiLayerNetwork network = alexnetModel(numLabels, channels);
        ModelSerializer.writeModel(network, getModelFile(), true);
    }

    public void train() throws IOException {
        init();
        int numLabels = labels.size();

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        FileSplit fileSplit = new FileSplit(dataDir.toFile(), NativeImageLoader.ALLOWED_FORMATS, random);
        int numExamples = toIntExact(fileSplit.length());
        BalancedPathFilter pathFilter = new BalancedPathFilter(random, labelMaker, numExamples, numLabels, 0);

        InputSplit[] inputSplit = fileSplit.sample(pathFilter, 1.0 - testFraction, testFraction);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];
        System.out.println("train data size: " + trainData.length());
        System.out.println("test data size: " + testData.length());

        ImageTransform flipTransform1 = new FlipImageTransform(random);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(random, 42);
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(flipTransform1, 0.9),
                new Pair<>(flipTransform2, 0.8),
                new Pair<>(warpTransform, 0.5));
        ImageTransform transform = new PipelineImageTransform(pipeline, false);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        File modelFile = getModelFile();
        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(modelFile);

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        // train
        {
            recordReader.initialize(trainData, transform);
            DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);

            for (int epoch = 0; epoch < epochs; epoch++) {
                System.out.println("Training epoch #" + epoch);
                network.fit(dataIter);
                System.out.println("score: " + network.score());
            }
        }

        // test
        {
            recordReader.initialize(testData);
            DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            Evaluation eval = network.evaluate(dataIter);
            System.out.println(eval.stats());
            System.out.println(eval.confusionToString());
        }

        ModelSerializer.writeModel(network, modelFile, true);
    }

    public void run(List<String> files) throws IOException {
        init();
        int numLabels = labels.size();

        File modelFile = getModelFile();
        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(modelFile);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels);

        ImageTransform transform = new ResizeImageTransform(width, height);

        CollectionInputSplit runData = new CollectionInputSplit(toURI(files));
        System.out.println("Data: " + runData.length());

        recordReader.initialize(runData);
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        //Evaluation evaluation = network.evaluate(dataIter);

        while (dataIter.hasNext()) {
            DataSet dataSet = dataIter.next();
            int[] prediction = network.predict(dataSet.getFeatures());
            System.out.println("PREDICTION " + Arrays.toString(prediction));
        }

    }

    private void init() {
        labels.addAll(loadLabels());
    }

    private List<String> loadLabels() {
        File[] files = dataDir.toFile().listFiles(File::isDirectory);
        return Arrays.stream(files)
                .filter(File::isDirectory)
                .map(File::getName)
                .filter(name -> !name.startsWith("_"))
                .filter(name -> !name.startsWith("."))
                .collect(Collectors.toList());
    }

    private List<URI> toURI(List<String> files) {
        List<URI> result = new ArrayList<>();

        for (String file : files) {
            result.add(new File(file).toURI());
        }

        return result;
    }

    public void detect(List<String> files) {

    }

    private File getModelFile() {
        return modelDir.resolve("model.dl4j").toFile();
    }

    private MultiLayerNetwork alexnetModel(int numLabels, int channels) {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/

        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .weightInit(WeightInit.DISTRIBUTION)
                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .l2(5 * 1e-4)
                .list()
                .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, maxPool("maxpool1", new int[]{3,3}))
                .layer(3, conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, maxPool("maxpool2", new int[]{3,3}))
                .layer(6, conv3x3("cnn3", 384, 0))
                .layer(7, conv3x3("cnn4", 384, nonZeroBias))
                .layer(8, conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, maxPool("maxpool3", new int[]{3,3}))
                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }

    private static void printHelp() {
        System.out.println("Usage: classify COMMAND [OPTIONS]");
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            printHelp();
            System.exit(1);
        }

        HelloDl4j hello = new HelloDl4j();

        String command = args[0];
        List<String> arguments = new ArrayList<>();
        for (int i = 1; i < args.length; i++) {
            if (args[i].startsWith("--")) {
                switch(args[i]) {
                    case "--data":
                        hello.dataDir = Paths.get(args[++i]);
                        break;
                    case "--model":
                        hello.modelDir = Paths.get(args[++i]);
                        break;
                    case "--width":
                        hello.width = Integer.parseInt(args[++i]);
                        break;
                    case "--height":
                        hello.height = Integer.parseInt(args[++i]);
                        break;
                    default:
                        System.out.println("Unknown option: " + args[i]);
                        System.exit(3);
                }
            } else {
                arguments.add(args[i]);
            }
        }

        switch (command) {
            case "create":
                hello.create();
                break;
            case "train":
                hello.train();
                break;
            case "test":
                hello.test();
                break;
            case "run":
                hello.run(arguments);
                break;
            default:
                System.out.println("Unknown command: " + command);
                System.exit(2);
        }
    }
}
