//
// Created by George Lancaster on 12/11/2018.
//
#include "header/calculations.h"
#include "header/hough.h"



vector<Rect> detect_dartboards(Mat image, CascadeClassifier model){
    vector<Rect> ret;
    vector<Rect> circles = hough_circle(image, 12, 40, 80);
    vector<Rect> boards;
    vector<Rect> det_boards;


    if(circles.size() != 0){
        for (Rect r : circles){
            Mat roi = Mat(image, r);
            model.detectMultiScale(roi, boards, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));
            if (boards.size() != 0){ // if a detection has been made
                Rect rect(boards[0].x + r.x, boards[0].y + r.y,boards[0].width, boards[0].height);
                det_boards.emplace_back(rect);
            }

        }
    }else{  // circles is empty
        model.detectMultiScale(image, det_boards, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));
    }
    if(det_boards.size() == 0){
        model.detectMultiScale(image, det_boards, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));
    }


    return det_boards;
}



/**
 * Calculate total f1 score for a data set
 */
double calculate_total_f1(vector<Mat> images, vector<vector<Rect>> ground_truth, CascadeClassifier model){

    // calculate the total tpr
    vector<vector<Rect>> detections;
    int tpr = 0;
    int total_detections = 0;
    int total_truth = 0;

    for (int i = 0; i < images.size(); i++){
        detections.emplace_back(detect(images[i], model));
        tpr += calculate_tpr(ground_truth[i], detections[i], images[i]);
    }

    // calculate total detections
    for (vector<Rect> detection : detections){
        total_detections += detection.size();
    }

    // calculate ground truth
    for (vector<Rect> truth : ground_truth){
        total_truth += truth.size();
    }


    double f1 = calculate_f1(tpr, total_detections, total_truth);

    cout << "\nf1 score for dataset: " << f1 << endl;

    return f1;
}

/**
 *
 * @param a: Rect
 * @param b: Rect
 * @return
 */
int overlap(Rect a, Rect b) {
    if ((a.x + a.width) <= b.x || a.x >= (b.x + b.width) || a.y >= (b.y + b.height) || (a.y + a.height) <= b.y) {
        return 0;
    }
    return 1;
}



/**
 * Calculate the ground truth
 * @param ground_truth: vector of Rects in the positions of ground truth
 * @param detections: vector containing all detections
 * @param image:
 * @return The number of correct detections
 */
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

                // if overlap is greater than 50%, add to correct detections
                if (overlap_percent >= 10){
                    // && rect_b.area() <= 2 * rect_a.area()) {
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

    //imshow("Correct detections", image);
    //waitKey(0);
    cout << "number of true positive detections: " << correct_detections.size() << endl;
    return (int) correct_detections.size();
}

/**
 * Calculate the f1 score
 * formula found on :
 * https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use
 * @param true_positives: number of true positives
 * @param total_detections: total number of detections
 * @param true_detections: the ground truth number of detections
 * @return f1 score
 */
double calculate_f1(int true_positives, int total_detections, int true_detections) {
    double false_positives = total_detections - true_positives;
    double precision = true_positives / (true_positives + false_positives);
    double false_negatives = true_detections - true_positives;
    double recall = true_positives / (true_positives + false_negatives);
    printf("Precision: %f\tRecall: %f\t\n",precision,recall);

    return 2 * ((precision * recall) / (precision + recall));
}

/**
 * Detect features from an image
 * @param image: input image
 * @param model: The classifier
 * @return result: Vector of all detections
 */
vector<Rect> detect(Mat image, CascadeClassifier model) {
    Mat gray_image;
    vector<Rect> result;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    model.detectMultiScale(gray_image, result, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));
    return result;
}
