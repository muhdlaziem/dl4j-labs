package io.skymind.labs.classification.kantopokemon;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

public class vgg16 {
    private static final int SEED = 5705;
    private static final int CLASS = 150;

    public static void main(String[] args) throws IOException {
        Path dataPath = Paths.get("pokemonclassification", "PokemonData");
        System.out.println(dataPath);

        FileSplit fileSplit = new FileSplit(new File(dataPath.toString()));
        System.out.println(fileSplit.toString());
        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();

        Random random = new Random(SEED);
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(random, BaseImageLoader.ALLOWED_FORMATS,labelGenerator);

        InputSplit[] trainTestSplit = fileSplit.sample(balancedPathFilter, 80,20);
        InputSplit trainData = trainTestSplit[0];
        InputSplit testData = trainTestSplit[1];

        ImageRecordReader trainRecordReader = new ImageRecordReader(255,255,3, labelGenerator);
        ImageRecordReader testRecordReader = new ImageRecordReader(255,255,3, labelGenerator);

        trainRecordReader.initialize(trainData);
        testRecordReader.initialize(testData);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, 32,1,150);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader, 32,1,150);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
//        System.out.println(vgg16.summary());

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-4))
                .seed(SEED)
                .build();

        ComputationGraph vgg16Transfer = new org.deeplearning4j.nn.transferlearning.TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("block5_pool") //"block5_pool" and below are frozen
                .nOutReplace("fc2",1024, WeightInit.XAVIER) //modify nOut of the "fc2" vertex
                .removeVertexAndConnections("predictions") //remove the final vertex and it's connections
                .addLayer("fc3",new DenseLayer
                        .Builder().activation(Activation.RELU).nIn(1024).nOut(256).build(),"fc2") //add in a new dense layer
                .addLayer("newpredictions",new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(256)
                        .nOut(CLASS)
                        .build(),"fc3") //add in a final output dense layer,
                // configurations on a new layer here will be override the finetune confs.
                // For eg. activation function will be softmax not RELU
                .setOutputs("newpredictions") //since we removed the output vertex and it's connections we need to specify outputs for the graph
                .build();
        System.out.println(vgg16Transfer.summary());

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);

        vgg16Transfer.setListeners(
                new StatsListener( statsStorage),
                new ScoreIterationListener(1)
        );
//        System.out.println(trainIter.next());
        vgg16Transfer.fit(trainIter, 10);
        Evaluation eval = vgg16Transfer.evaluate(testIter);
        System.out.println(eval.stats());
    }
}
