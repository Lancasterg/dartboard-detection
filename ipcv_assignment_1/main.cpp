#include <iostream>
#include <opencv2/opencv.hpp>
#include "header/main.h"
#include "header/load_faces.h"
#include "header/load_darts.h"
#include "header/calculations.h"

/** Namespace declaration **/
using namespace std;
using namespace cv;


/** Global variables */
String cascade_name = "../frontalface.xml";
String cascade_dart_name = "../training_data/dartcascade/cascade.xml";
CascadeClassifier cascade;



/** @function main */
int main(int argc, const char **argv) {
    CascadeClassifier model;

    if (!model.load(cascade_name)) {
        printf("--(!)Error loading\n");
        return 1;
    };

    vector<Mat> images = load_test_images();
    vector<vector<Rect>> true_labels = load_face_labels();

    double f1 = calculate_total_f1(images, true_labels, model);






    //label_faces();
    //label_darts();
    //task_one_a();
    //task_one_b();
    //task_two();
    //task_one_b();
    //task_two_b();

    return 0;
}

void task_two() {
    // 1. Load the Strong Classifier in a structure called `Cascade'
    if (!cascade.load(cascade_dart_name)) {
        printf("--(!)Error loading\n");
        return;
    };

    // 2. Load in the images from file
    Mat frames[16];
    char input_str[18];
    for (int i = 0; i < 15; i++) {
        sprintf(input_str, "../img/dart%d.jpg", i);
        frames[i] = imread(input_str);
    }


    for (int i = 0; i < 15; i++) {
        char output[50];
        sprintf(output, "../task_2_detected/detected%d.jpg", i);
        // 3. Detect Faces and Display Result
        detectAndDisplay(frames[i]);
        // 4. Save Result Image
        imwrite(output, frames[i]);
    }
}

void task_one_a() {

    // Task 1.
    // Create array of images.
    Mat frames[5];
    frames[0] = imread("../img/dart4.jpg");
    frames[1] = imread("../img/dart5.jpg");
    frames[2] = imread("../img/dart13.jpg");
    frames[3] = imread("../img/dart14.jpg");
    frames[4] = imread("../img/dart15.jpg");

    // 2. Load the Strong Classifier in a structure called `Cascade'
    if (!cascade.load(cascade_name)) {
        printf("--(!)Error loading\n");
        return;
    };


    //auto true_5 = load_faces_5();
    //auto true_15 = load_faces_15();

    //auto rects_5 = detect(frames[1], cascade);
    //auto rects_15 = detect(frames[4], cascade);

    /*
    for (int i = 0; i < 5; i++) {
        char output[50];
        sprintf(output, "../detected%d.jpg", i);
        // 3. Detect Faces and Display Result
        detectAndDisplay(frames[i]);
        // 4. Save Result Image
        imwrite(output, frames[i]);
    }
    */
}


void get_dataset_f1(){

}

/** Task 1b, calculate the TPR for images 5 and 15 when detecting for faces */
void task_one_b() {
    // read in images
    Mat darts5 = imread("../img/dart5.jpg");
    Mat darts15 = imread("../img/dart15.jpg");

    vector<Rect> darts5_truth = load_faces_5();
    vector<Rect> darts15_truth = load_faces_15();

    // load classifier
    if (!cascade.load(cascade_name)) {
        printf("--(!)Error loading\n");
        return;
    };

    // detect faces using the cascade classifier
    vector<Rect> darts_5_detection = detect(darts5, cascade);
    vector<Rect> darts_15_detection = detect(darts15, cascade);

    // calulate tpr and f1 scores
    int darts5tpr = calculate_tpr(darts5_truth, darts_5_detection, darts5);
    double f1 = calculate_f1(darts5tpr, (int) darts_5_detection.size(), (int) darts5_truth.size());

    int darts15tpr = calculate_tpr(darts15_truth, darts_15_detection, darts15);
    double f1_15 = calculate_f1(darts15tpr, (int) darts_15_detection.size(), (int) darts15_truth.size());

    cout << "darts5 tpr: " << darts5tpr << endl;
    cout << "darts 5 f1 score: " << f1 << endl;

    cout << "darts15 tpr: " << darts15tpr << endl;
    cout << "darts 15 f1 score: " << f1_15 << endl;

}

/** Test the dartboard detector on all example images **/
void task_two_b(){
    vector<Mat> frames;
    vector<vector<Rect>> rects;
    vector<int> tpr;
    vector<double> f1_scores;
    vector<vector<Rect>> true_labels;
    char input_str[18];

    // 1. Load the Strong Classifier in a structure called `Cascade'
    if (!cascade.load(cascade_dart_name)) {
        printf("--(!)Error loading\n");
        return;
    };

    true_labels = load_dart_labels();

    for (int i = 0; i < 16; i++) {
        sprintf(input_str, "../img/dart%d.jpg", i);
        frames.emplace_back(imread(input_str));
    }

    // detect all rects
    for (Mat m : frames){
        rects.emplace_back(detect(m, cascade));
    }

    for (int i = 0; i < 16; i++){
        for (int j = 0; j < rects[i].size(); j++){
            rectangle(frames[i], Point(rects[i][j].x, rects[i][j].y),
                      Point(rects[i][j].x + rects[i][j].width, rects[i][j].y + rects[i][j].height), Scalar(0, 255, 0), 2);

        }
        imshow("frames", frames[i]);
        waitKey(0);

    }

    // calculate tpr
    for (int i = 0; i < 16; i++){
        tpr.emplace_back(calculate_tpr(true_labels.at(i), rects.at(i), frames.at(i)));
    }

    // calculate f1 score for each
    for (int i = 0; i < 16; i++){
        f1_scores.emplace_back(calculate_f1(tpr.at(i),rects.at(i).size(),true_labels.at(i).size()));
        cout << "f1_score: " << f1_scores.at(i) << endl;
    }
}


/**
 * @function detectAndDisplay
 */
void detectAndDisplay(Mat frame) {
    std::vector<Rect> faces;
    Mat frame_gray;

    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

    // 3. Print number of Faces found
    std::cout << faces.size() << std::endl;

    // 4. Draw box around faces found
    for (int i = 0; i < faces.size(); i++) {
        rectangle(frame, Point(faces[i].x, faces[i].y),
                  Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 2);
    }

}



