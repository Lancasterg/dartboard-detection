//
// Created by George Lancaster on 12/11/2018.
//
#include "header/calculations.h"
#include "header/hough.h"
#include "header/util.h"

vector<Rect> exlude_intersect(vector<Rect> boards) {
    vector<Rect> result;
    vector<Rect> toExclude;

    for (Rect board : boards) {
        for (Rect b : boards) {
            if (board.x == b.x && board.y == b.y && board.width == b.width && board.height == b.height) {
                continue;
            }

            if ((board & b).area() > 0) {
                if (b.area() < board.area()) {
                    toExclude.insert(toExclude.end(), b);
                }
            }

        }
    }


    for (Rect board : boards) {
        bool exclude = false;
        for (Rect b : toExclude) {
            if (board.x == b.x && board.y == b.y && board.width == b.width && board.height == b.height) {
                exclude = true;
                break;
            }
        }

        if (!exclude) {
            result.insert(result.end(), board);
        }
    }

    return result;
}

vector<Rect> detect_dartboards(Mat image, CascadeClassifier model) {
    vector<Rect> ret;
    vector<Rect> circles = concentric_intersection(image);
    vector<Rect> boards;
    vector<Rect> det_boards;


    if (!circles.empty()) {
        for (Rect r : circles) {
            Mat roi = Mat(image, r);
//            imshow("roi", roi);
//            waitKey(0);
            model.detectMultiScale(roi, boards, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));
            if (boards.size() != 0) { // if a detection has been made

                // find average rect area
                int avg_x = 0;
                int avg_y = 0;
                int avg_height = 0;
                int avg_width = 0;
                for (Rect b : boards) {
                    avg_x += b.x;
                    avg_y += b.y;
                    avg_height += b.height;
                    avg_width += b.width;
                }
                avg_x /= boards.size();
                avg_y /= boards.size();
                avg_height /= boards.size();
                avg_width /= boards.size();

                // add the rects together


                Rect rect(avg_x - 10 + r.x, avg_y + r.y - 10, avg_width + 10, avg_height + 10);
                det_boards.emplace_back(rect);
            }

        }
    } else {  // circles is empty
        model.detectMultiScale(image, det_boards, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));
    }
    if (det_boards.size() == 0) {
        model.detectMultiScale(image, det_boards, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));
    }

    // filter by line intersection
    det_boards = line_intersection(image, det_boards);

    det_boards = exlude_intersect(det_boards);

    det_boards = template_matching(image, det_boards);

    return det_boards;
}


/**
 * Calculate total f1 score for a data set
 */
double calculate_total_f1(vector<Mat> images, vector<vector<Rect>> ground_truth, CascadeClassifier model) {

    // calculate the total tpr
    vector<vector<Rect>> detections;
    int tpr = 0;
    int total_detections = 0;
    int total_truth = 0;

    for (int i = 0; i < images.size(); i++) {
        detections.emplace_back(detect(images[i], model));
        tpr += calculate_tpr(ground_truth[i], detections[i], images[i]);
    }

    // calculate total detections
    for (vector<Rect> detection : detections) {
        total_detections += detection.size();
    }

    // calculate ground truth
    for (vector<Rect> truth : ground_truth) {
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
                if (overlap_percent >= 10) {
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
    double f1 = 2 * ((precision * recall) / (precision + recall));
    printf("f1 score: %f\tprecision: %f\t recall: %f\t\n\n", f1, precision, recall);
    return f1;
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
