//
// Created by George Lancaster on 12/11/2018.
//

#ifndef OPENCV_CALCULATIONS_H
#define OPENCV_CALCULATIONS_H
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

double calculate_f1(int true_positives, int total_detections, int true_detections);
vector<Rect> detect(Mat image, CascadeClassifier model);
int calculate_tpr(vector<Rect> ground_truth, vector<Rect> detections, Mat image);
int overlap(Rect a, Rect b);
void detectAndDisplay(Mat frame);
double calculate_total_f1(vector<Mat> images, vector<vector<Rect>> ground_truth, CascadeClassifier model);

#endif //OPENCV_CALCULATIONS_H
