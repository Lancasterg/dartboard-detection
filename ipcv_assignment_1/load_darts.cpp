//
// Created by George Lancaster on 08/11/2018.
//
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <vector>
#include <fstream>
#include "header/load_darts.h"
#include "header/load_faces.h"

using namespace std;
using namespace cv;



/**
 * Label all faces and save them to ../labelled_dartboards
 **/
void label_darts(){
    vector<vector<Rect>> darts_labels = load_dart_labels();
    vector<Mat> images = load_test_images();


    for(int i = 0; i < darts_labels.size(); i++){
        char output[50];
        Mat img = images.at(i);
        sprintf(output, "../labelled_dartboards/labelleddartboards%d.jpg", i);
        for(int j = 0; j < darts_labels.at(i).size(); j++){
            rectangle(images.at(i), Point(darts_labels.at(i).at(j).x, darts_labels.at(i).at(j).y), Point(darts_labels.at(i).at(j).x + darts_labels.at(i).at(j).width, darts_labels.at(i).at(j).y + darts_labels.at(i).at(j).height), Scalar( 0, 255, 0 ), 2);
        }
        imwrite(output, img);
    }
}

/**
 * Load all ground_truth dartboard labels
 * @return vector containing all ground truth dartboard labels
 */
vector<vector<Rect>> load_dart_labels(){
    vector<vector<Rect>> result;
    result.emplace_back(load_darts_0());
    result.emplace_back(load_darts_1());
    result.emplace_back(load_darts_2());
    result.emplace_back(load_darts_3());
    result.emplace_back(load_darts_4());
    result.emplace_back(load_darts_5());
    result.emplace_back(load_darts_6());
    result.emplace_back(load_darts_7());
    result.emplace_back(load_darts_8());
    result.emplace_back(load_darts_9());
    result.emplace_back(load_darts_10());
    result.emplace_back(load_darts_11());
    result.emplace_back(load_darts_12());
    result.emplace_back(load_darts_13());
    result.emplace_back(load_darts_14());
    result.emplace_back(load_darts_15());

    return result;
}
vector<Rect> load_darts_0(){
    vector<Rect> res;
    res.emplace_back(Rect(441,25,174,175));
    return res;

}
vector<Rect> load_darts_1(){
    vector<Rect> res;
    res.emplace_back(Rect(183,121,230,227));
    return res;

}
vector<Rect> load_darts_2(){
    vector<Rect> res;
    res.emplace_back(Rect(99,93,99,101));
    return res;

}
vector<Rect> load_darts_3(){
    vector<Rect> res;
    res.emplace_back(Rect(321,145,72,80));
    return res;

}
vector<Rect> load_darts_4(){
    vector<Rect> res;
    res.emplace_back(Rect(170,98,180,166));
    return res;

}
vector<Rect> load_darts_5(){
    vector<Rect> res;
    res.emplace_back(Rect(434,146,95,100));
    return res;
}
vector<Rect> load_darts_6(){
    vector<Rect> res;
    res.emplace_back(Rect(203, 115, 78, 74));
    return res;

}
vector<Rect> load_darts_7(){
    vector<Rect> res;
    res.emplace_back(Rect(247,159,100,152));
    return res;

}
vector<Rect> load_darts_8(){
    vector<Rect> res;
    res.emplace_back(Rect(65,252,65,98));
    res.emplace_back(Rect(835,214,134,150));
    return res;

}
vector<Rect> load_darts_9(){
    vector<Rect> res;
    res.emplace_back(Rect(194,18,255,281));
    return res;

}
vector<Rect> load_darts_10(){
    vector<Rect> res;
    res.emplace_back(Rect(72,95,127,133));
    res.emplace_back(Rect(576,115,76,111));
    res.emplace_back(Rect(900,130,76,100));
    return res;

}
vector<Rect> load_darts_11(){
    vector<Rect> res;
    res.emplace_back(Rect(171,97,71,55));
    return res;

}
vector<Rect> load_darts_12(){
    vector<Rect> res;
    res.emplace_back(Rect(151,62,75,170));
    return res;

}
vector<Rect> load_darts_13(){
    vector<Rect> res;
    res.emplace_back(Rect(261,107,155,161));
    return res;

}
vector<Rect> load_darts_14(){
    vector<Rect> res;
    res.emplace_back(Rect(99,95,166,152));
    res.emplace_back(Rect(972,83,165,156));
    return res;

}
vector<Rect> load_darts_15(){
    vector<Rect> res;
    res.emplace_back(Rect(159,62,124,131));
    return res;

}


