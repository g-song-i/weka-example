package com.company;

import java.io.*;
import java.util.*;
import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.classifiers.Evaluation;

/*
 * @author Gwak song-i
 * Email : rkdlem1613@naver.com
 * Github link : https://github.com/g-song-i/weka-example
 */

public class Main {

    public static void main(String[] args) {

        try {
            Main object = new Main();
            object.J48Tree();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }
    public void J48Tree() throws Exception{

        // 1) data load
        Instances data = new Instances(
                new BufferedReader(
                        new FileReader("C:\\Program Files\\Weka-3-9-4\\data\\diabetes.arff")
                ));

        /*
        int trainSize = (int)Math.round(data.numInstances() * percent / 100);
        int testSize = data.numInstances() - trainSize();
        data.randomize(new java.util.Random(seed));

        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, traingSize, testSize);
         */

        // 2) class assigner
        //train.setClassIndex(train.numAttributes() - 1);
        //test.setClassIndex(test.numAttributes() - 1);
        data.setClassIndex(data.numAttributes() - 1);

        // 3) cross validate setting
        Evaluation eval = new Evaluation(data);
        Classifier model = new J48();
        eval.crossValidateModel(model, data, 10, new Random(1));

        // 4) random forest run
        model.buildClassifier(data);

        // 6) print result text
        System.out.println("결과 : " + eval.toSummaryString());

    }
}
