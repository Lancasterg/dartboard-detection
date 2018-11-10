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
#include "header/load_faces.h"

using namespace std;
using namespace cv;


/** Label all faces and save them to ../labelled_faces **/
void label_faces(){
    vector<vector<Rect>> faces_labels = load_face_labels();
    vector<Mat> images = load_test_images();


    for(int i = 0; i < faces_labels.size(); i++){
        char output[50];
        Mat img = images.at(i);
        sprintf(output, "../labelled_faces/labelledfaces%d.jpg", i);
        for(int j = 0; j < faces_labels.at(i).size(); j++){
            rectangle(images.at(i), Point(faces_labels.at(i).at(j).x, faces_labels.at(i).at(j).y), Point(faces_labels.at(i).at(j).x + faces_labels.at(i).at(j).width, faces_labels.at(i).at(j).y + faces_labels.at(i).at(j).height), Scalar( 0, 255, 0 ), 2);
        }
        imwrite(output, img);
    }
}


vector<Mat> load_test_images(){
    vector<Mat> res;
    for(int i = 0; i < 16; i++) {
        char input[50];
        sprintf(input, "../img/dart%d.jpg", i);
        res.emplace_back(imread(input));
    }
    return res;
}

vector<vector<Rect>> load_face_labels(){
    vector<vector<Rect>> result;
    result.emplace_back(load_faces_0());
    result.emplace_back(load_faces_1());
    result.emplace_back(load_faces_2());
    result.emplace_back(load_faces_3());
    result.emplace_back(load_faces_4());
    result.emplace_back(load_faces_5());
    result.emplace_back(load_faces_6());
    result.emplace_back(load_faces_7());
    result.emplace_back(load_faces_8());
    result.emplace_back(load_faces_9());
    result.emplace_back(load_faces_10());
    result.emplace_back(load_faces_11());
    result.emplace_back(load_faces_12());
    result.emplace_back(load_faces_13());
    result.emplace_back(load_faces_14());
    result.emplace_back(load_faces_15());

    return result;
}
vector<Rect> load_faces_0(){
    vector<Rect> res;
    res.emplace_back(Rect(200,200,63,96));
    return res;

}
vector<Rect> load_faces_1(){
    vector<Rect> res;
    res.emplace_back(Rect(0, 0, 0, 0));
    return res;

}
vector<Rect> load_faces_2(){
    vector<Rect> res;
    res.emplace_back(Rect(0,0,0,0));
    return res;

}
vector<Rect> load_faces_3(){
    vector<Rect> res;
    res.emplace_back(Rect(0,0,0,0));
    return res;

}
vector<Rect> load_faces_4(){
    vector<Rect> res;
    res.emplace_back(Rect(353,111,117,154));
    return res;

}
vector<Rect> load_faces_5(){
    vector<Rect> res;
    res.emplace_back(Rect(40,261,70,70));
    res.emplace_back(Rect(72,146,41,61));
    res.emplace_back(Rect(197,227,51,56));
    res.emplace_back(Rect(296,254,50,58));
    res.emplace_back(Rect(252,175,47,54));
    res.emplace_back(Rect(434,244,49,61));
    res.emplace_back(Rect(563,258,51,55));
    res.emplace_back(Rect(513,184,56,59));
    res.emplace_back(Rect(648,199,55,46));
    res.emplace_back(Rect(678,252,65,56));
    res.emplace_back(Rect(373,194,66,58));
    return res;
}
vector<Rect> load_faces_6(){
    vector<Rect> res;
    res.emplace_back(Rect(287, 109, 38, 48));
    return res;

}
vector<Rect> load_faces_7(){
    vector<Rect> res;
    res.emplace_back(Rect(348,194,78,90));
    return res;

}
vector<Rect> load_faces_8(){
    vector<Rect> res;
    res.emplace_back(Rect(173,285,60,70));
    return res;

}
vector<Rect> load_faces_9(){
    vector<Rect> res;
    res.emplace_back(Rect(87,202,143,109));
    return res;

}
vector<Rect> load_faces_10(){
    vector<Rect> res;
    res.emplace_back(Rect(0,0,0,0));
    return res;

}
vector<Rect> load_faces_11(){
    vector<Rect> res;
    res.emplace_back(Rect(406,50,45,73));
    res.emplace_back(Rect(322,73,62,74));
    return res;

}
vector<Rect> load_faces_12(){
    vector<Rect> res;
    res.emplace_back(Rect(0,0,0,0));
    return res;

}
vector<Rect> load_faces_13(){
    vector<Rect> res;
    res.emplace_back(Rect(421,125,106,131));
    return res;

}
vector<Rect> load_faces_14(){
    vector<Rect> res;
    res.emplace_back(Rect(466,214,82,109));
    res.emplace_back(Rect(717,195,108,100));
    return res;

}
vector<Rect> load_faces_15(){
    vector<Rect> res;
    res.emplace_back(Rect(369,102,78,95));
    res.emplace_back(Rect(540,125,107,97));
    res.emplace_back(Rect(69,130, 54,83));
    return res;

}


