//
// Created by George Lancaster on 12/11/2018.
//

#ifndef OPENCV_MAIN_H
#define OPENCV_MAIN_H
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

void detectAndDisplay(Mat frame);

void task_one_a();

void task_two();

void task_one_b();

void label_faces();

void task_two();

void task_two_b();

void task_three(char *imageName);


#endif //OPENCV_MAIN_H
