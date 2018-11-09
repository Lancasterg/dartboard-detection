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
void task_one();
void task_two();
Mat* load_images();
void label_faces();

/** Global variables */
String cascade_name = "../frontalface.xml";
String cascade_dart_name = "../training_data/dartcascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{

    label_faces();
    //label_darts();
    //task_one();
    //task_two();
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

void task_one()
{
    // first number is the image
    vector<vector<Rect>> imageRects;
    imageRects.at(0).at(0).x = 1;
    imageRects.at(0).at(0).y = 1;



    // need to fill rects up with x, y, w ,h
    vector<Rect> rects;

    //cout << get<0>(a) << endl;



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



