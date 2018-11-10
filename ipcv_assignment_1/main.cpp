#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "header/load_faces.h"
#include "header/load_darts.h"


using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

void task_one_a();

void task_two();

void task_one_b();

Mat *load_images();

void label_faces();

double calculate_f1(int true_positives, int total_detections, int true_detections);

vector<Rect> detect(Mat image, CascadeClassifier model);

int calculate_tpr(vector<Rect> ground_truth, vector<Rect> detections, Mat image);

/** Global variables */
String cascade_name = "../frontalface.xml";
String cascade_dart_name = "../training_data/dartcascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main(int argc, const char **argv) {
    //label_faces();
    //label_darts();
    //task_one_a();
    //task_two();
    task_one_b();
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


    for (int i = 0; i < 5; i++) {
        char output[50];
        sprintf(output, "../detected%d.jpg", i);
        // 3. Detect Faces and Display Result
        detectAndDisplay(frames[i]);
        // 4. Save Result Image
        imwrite(output, frames[i]);
    }
}

/** Task 1b, calculate the TPR for images 5 and 15 when detecting for faces */
void task_one_b() {
    // read in images
    Mat darts5 = imread("../img/dart5.jpg");
    Mat darts15 = imread("../img/dart15.jpg");
    vector<Rect> darts5_truth = load_faces_5();

    // load classifier
    if (!cascade.load(cascade_name)) {
        printf("--(!)Error loading\n");
        return;
    };

    // detect faces using the cascade classifier
    vector<Rect> darts_5_detection = detect(darts5, cascade);

    // calulate tpr and f1 scores
    int darts5tpr = calculate_tpr(darts5_truth, darts_5_detection, darts5);
    double f1 = calculate_f1(darts5tpr, (int) darts_5_detection.size(), (int) darts5_truth.size());
    cout << "f1 score: " << f1 << endl;
}

/** find if two rects overlap **/
int overlap(Rect a, Rect b) {
    if ((a.x + a.width) <= b.x || a.x >= (b.x + b.width) || a.y >= (b.y + b.height) || (a.y + a.height) <= b.y) {
        return 0;
    }
    return 1;
}


/** calculate the true positive rate **/
int calculate_tpr(vector<Rect> ground_truth, vector<Rect> detections, Mat image) {

    vector<Rect> correct_detections;

    for (Rect rect_a : ground_truth) {
        for (Rect rect_b: detections) {

            // if no overlap, continue to next iteration
            if (!overlap(rect_a, rect_b)) {
                continue;
            } else {

                // Calculate the overlapping area
                int width = min(rect_a.x + rect_a.width, rect_b.x + rect_b.width) - max(rect_a.x, rect_b.x);
                int height = min(rect_a.y + rect_a.height, rect_b.y + rect_b.height) - max(rect_a.y, rect_b.y);;
                double overlap = width * height;
                double overlap_percent = (overlap / rect_a.area()) * 100;

                // if overlap is greater than 75%, add to correct detections
                if (overlap_percent >= 60 && rect_b.area() <= 2 * rect_a.area()) {
                    correct_detections.emplace_back(rect_b);
                    break;
                }
            }
        }
    }


    cout << "True detected rects\nx\ty" << endl;
    for (Rect r : correct_detections) {
        cout << r.x << "\t" << r.y << endl;
        rectangle(image, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 255, 0), 2);

    }

    imshow("Correct detections", image);
    waitKey(0);
    cout << "number of true positive detections: " << correct_detections.size() << endl;
    return (int) correct_detections.size();
}

/** Calculate the f1 score
 * https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use **/
double calculate_f1(int true_positives, int total_detections, int true_detections) {
    double false_positives = total_detections - true_positives;
    double precision = true_positives / (true_positives + false_positives);
    double false_negatives = true_detections - true_positives;
    double recall = true_positives / (true_positives + false_negatives);
    return 2 * ((precision * recall) / (precision + recall));
}


vector<Rect> detect(Mat image, CascadeClassifier model) {
    Mat gray_image;
    vector<Rect> result;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    model.detectMultiScale(gray_image, result, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));
    return result;
}


/** @function detectAndDisplay */
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



