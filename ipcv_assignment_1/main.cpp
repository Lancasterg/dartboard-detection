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
void detectAndDisplay( Mat frame );
void task_one_a();
void task_two();
void task_one_b();
Mat* load_images();
void label_faces();
vector<Rect> detect(Mat image, CascadeClassifier model);
int calculate_tpr(vector<Rect> ground_truth, vector<Rect> detections);

/** Global variables */
String cascade_name = "../frontalface.xml";
String cascade_dart_name = "../training_data/dartcascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
    //label_faces();
    //label_darts();
    //task_one_a();
    //task_two();
    task_one_b();




    return 0;
}

void task_two(){
    // 1. Load the Strong Classifier in a structure called `Cascade'
    if( !cascade.load( cascade_dart_name ) ){ printf("--(!)Error loading\n"); return; };

    // 2. Load in the images from file
    Mat frames[16];
    char input_str[18];
    for (int i = 0; i < 15; i++){
        sprintf(input_str,"../img/dart%d.jpg", i);
        frames[i] = imread(input_str);
    }


    for(int i = 0; i < 15; i++) {
        char output[50];
        sprintf(output, "../task_2_detected/detected%d.jpg", i);
        // 3. Detect Faces and Display Result
        detectAndDisplay(frames[i]);
        // 4. Save Result Image
        imwrite(output, frames[i]);
    }
}

void task_one_a()
{

    // Task 1.
    // Create array of images.
    Mat frames[5];
    frames[0] = imread("../img/dart4.jpg");
    frames[1] = imread("../img/dart5.jpg");
    frames[2] = imread("../img/dart13.jpg");
    frames[3] = imread("../img/dart14.jpg");
    frames[4] = imread("../img/dart15.jpg");

    // 2. Load the Strong Classifier in a structure called `Cascade'
    if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return; };


    for(int i = 0; i < 5; i++) {
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
    Mat darts5 = imread("../img/dart5.jpg");
    Mat darts15 = imread("../img/dart15.jpg");
    vector<Rect> darts5_truth = load_faces_5();
    vector<Rect> darts15_truth = load_faces_15();

    if (!cascade.load(cascade_name)) {
        printf("--(!)Error loading\n");
        return;
    };
    vector<Rect> darts_5_detection = detect(darts5, cascade);

    for( int i = 0; i < darts_5_detection.size(); i++ )
    {
        rectangle(darts5, Point(darts_5_detection[i].x, darts_5_detection[i].y), Point(darts_5_detection[i].x + darts_5_detection[i].width, darts_5_detection[i].y + darts_5_detection[i].height), Scalar( 0, 255, 0 ), 2);
    }

    //imshow("abc", darts5);
    //waitKey(0);


    int darts5tpr = calculate_tpr(darts5_truth, darts_5_detection);

    vector<Rect> darts_15_detection = detect(darts15, cascade);
    //int darts15tpr = calculate_tpr(darts15_truth,darts_15_detection);


    //cout << darts5tpr << endl;
    //cout << darts15tpr << endl;



}

int calculate_tpr(vector<Rect> ground_truth, vector<Rect> detections){
    int tp = 0;
    vector<Rect> correct_detections;
    for(int j = 0; j < ground_truth.size(); j++ ){
        for(int i = 0; i < detections.size(); i++ ){
            if (((detections.at(i).area() & ground_truth.at(j).area()) < 0)){
                continue; }
            else {

                // Calculate the overlapping area
                int width = min(ground_truth.at(j).x+ground_truth.at(j).width,detections.at(i).x+detections.at(i).width) - max(ground_truth.at(j).x, detections.at(i).x);
                int height = min(ground_truth.at(j).y+ground_truth.at(j).height,detections.at(i).y+detections.at(i).height) - max(ground_truth.at(j).y,detections.at(i).y);;
                double overlap = width*height;
                double perc = (overlap/ground_truth.at(j).area()) * 100;

                if (perc >= 0){
                    if (detections[i].area() <= 2*ground_truth[j].area() ){
                        correct_detections.emplace_back(detections[i]);
                        tp++;
                        break;
                    }
                }
            }
        }
    }
    Mat darts5 = imread("../img/dart5.jpg");
    for( int i = 0; i < correct_detections.size(); i++ ) {
        rectangle(darts5, Point(correct_detections[i].x, correct_detections[i].y), Point(correct_detections[i].x + correct_detections[i].width, correct_detections[i].y + correct_detections[i].height), Scalar( 0, 255, 0 ), 2);
    }

    imshow("Correct detections", darts5);
    waitKey(0);

    cout << "detections: " << correct_detections.size() << endl;
    return tp;
}


vector<Rect> detect(Mat image, CascadeClassifier model){
    Mat gray_image;
    vector<Rect> result;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    model.detectMultiScale( gray_image, result, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
    return result;
}







/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;

    // 1. Prepare Image by turning it into Grayscale and normalising lighting
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    // 2. Perform Viola-Jones Object Detection
    cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

    // 3. Print number of Faces found
    std::cout << faces.size() << std::endl;

    // 4. Draw box around faces found
    for( int i = 0; i < faces.size(); i++ )
    {
        rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
    }

}



