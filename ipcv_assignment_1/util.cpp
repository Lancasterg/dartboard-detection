//
// Created by Jeaung on 11/16/18.
//

#include "header/hough.h"
#include "header/util.h"
#include "opencv2/xfeatures2d.hpp"

using namespace cv::xfeatures2d;

vector<Rect> sliding_window_classification(Mat &src, CascadeClassifier model) {
    int min_window = 40;
    int max_window = src.rows;
    vector<Rect> ret;
    vector<Rect> detections;
//    for (int window_size = min_window; window_size < max_window; window_size++){
//
//        for (int i = 0; i < src.rows - window_size; i+= 0.5 * window_size){
//            for (int j = 0; j < src.cols - window_size; j+= 0.5 * window_size){
//
//                vector<Rect> cur_det;
//                Mat dst_roi;
//                src(Rect(j,i,window_size, window_size)).copyTo(dst_roi);
//                model.detectMultiScale(dst_roi, cur_det, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(min_window, min_window), Size(window_size, window_size));
//
//                if (cur_det.size() != 0) { // if a detection has been made
//                    Rect rect(cur_det[0].x + src.rows, cur_det[0].y + src.cols, cur_det[0].width, cur_det[0].height);
//                    detections.emplace_back(rect);
//                }
//            }
//        }
//    }

//
//
//    model.detectMultiScale(src, detections, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(cur_window, cur_window), Size(cur_wind, 250));
//
//    for (Rect r: detections){
//        rectangle( src, r, Scalar(255,0,0), 2, 8, 0 );
//    }
//    imshow("dst_roi", src);
//    waitKey(0);

    return ret;
}

vector<Vec2f> getCentral(Mat sub, vector<Vec2f> lines) {
    float diag = sqrt(pow(sub.rows, 2) + pow(sub.cols, 2));

    vector<Vec2f> centralLines;
    for (Vec2f l : lines) {
        float rho = l.val[0];
        float theta = l.val[1];

        Point2f o1, p1;
        float a = cos(theta), b = sin(theta);
        float x0 = a * rho, y0 = b * rho;
        o1.x = cvRound(x0 + 1000 * (-b));
        o1.y = cvRound(y0 + 1000 * (a));
        p1.x = cvRound(x0 - 1000 * (-b));
        p1.y = cvRound(y0 - 1000 * (a));

        float factor = 1 / 4.3;

        if (rho < diag * factor || rho > diag * (1 - factor)) {
            continue;
        } else {
            line(sub, o1, p1, Scalar(0, 0, 255), 1, CV_AA);
            centralLines.insert(centralLines.end(), l);
        }
    }

//    imshow("detected lines", sub);
//    waitKey(0);

    return centralLines;
}

bool diverseDegree(vector<Vec2f> lines) {
    float totalDifference = 0;
    int n = 0;
    for (Vec2f line : lines) {
        for (Vec2f l : lines) {
            if (fabs(line.val[0] - l.val[0]) < 1e-6 && fabs(line.val[1] - l.val[1]) < 1e-6) {
                continue;
            } else {
                n++;
                totalDifference += fabs(line.val[1] * 180 / M_PI - l.val[1] * 180 / M_PI);
            }
        }
    }

    float avg = totalDifference / n;

    cout << "degree difference avg" << avg << endl;

    return avg > 8;
}

vector<Rect> line_intersection(const Mat &src, vector<Rect> &circles) {
    vector<Rect> result;

    for (vector<Rect>::iterator it = circles.begin(); it != circles.end(); it++) {
        printf("detected rectangles x %d  y %d width %d height %d\n", it->x, it->y, it->width, it->height);

        Range rows(it->y, it->y + it->height);
        Range cols(it->x, it->x + it->width);

        Mat sub = src(rows, cols).clone();

//        imshow("aaa", sub);
//        waitKey(0);

        // detect lines in a sub-region
        vector<Vec2f> lines = hough_line(sub, it->width / 3.4, 25);

        vector<Vec2f> centralLines = getCentral(sub, lines);

        if (!diverseDegree(centralLines)) {
            cout << "degree difference too small" << endl;
            continue;
        }

        if (centralLines.size() < MIN_LINES) { // only one line detected, it's not a dartboard
            continue;
        }

        // find all points of intersection in a sub-region
        vector<Point2f> intersections;

        for (vector<Vec2f>::iterator lineIt1 = centralLines.begin(); lineIt1 != centralLines.end(); lineIt1++) {
            float rho1 = lineIt1->val[0];
            float theta1 = lineIt1->val[1];

//            printf("line rho1 %f  theta1 %f\n", rho1, theta1);

            Point2f o1, p1;
            float a = cos(theta1), b = sin(theta1);
            float x0 = a * rho1, y0 = b * rho1;
            o1.x = cvRound(x0 + 1000 * (-b));
            o1.y = cvRound(y0 + 1000 * (a));
            p1.x = cvRound(x0 - 1000 * (-b));
            p1.y = cvRound(y0 - 1000 * (a));

            for (vector<Vec2f>::iterator lineIt2 = centralLines.begin(); lineIt2 != centralLines.end(); lineIt2++) {
                float rho2 = lineIt2->val[0];
                float theta2 = lineIt2->val[1];

//                printf("line rho2 %f  theta2 %f\n", rho2, theta2);

                // exclude self
                if (fabs(rho1 - rho2) < 1e-4 && fabs(theta1 - theta2) < 1e-4) {
                    continue;
                }

                Point2f o2, p2;
                float a = cos(theta2), b = sin(theta2);
                float x0 = a * rho2, y0 = b * rho2;
                o2.x = cvRound(x0 + 1000 * (-b));
                o2.y = cvRound(y0 + 1000 * (a));
                p2.x = cvRound(x0 - 1000 * (-b));
                p2.y = cvRound(y0 - 1000 * (a));

                Point2f x = o2 - o1;
                Point2f d1 = p1 - o1;
                Point2f d2 = p2 - o2;
                float cross = d1.x * d2.y - d1.y * d2.x;
                if (abs(cross) > 1e-8) {
                    double t1 = (x.x * d2.y - x.y * d2.x) / cross;
                    Point2f r = o1 + d1 * t1;
                    if (r.x > 0 && r.x < sub.cols && r.y > 0 && r.y < sub.rows) {
                        intersections.insert(intersections.end(), r);
                    }
                }
            }
        }

        if (intersections.empty()) {
            continue;
        }

        Rect2f center;
        center.x = (float) sub.cols * 2.95 / 10;
        center.y = (float) sub.rows * 2.95 / 10;
        center.width = (float) sub.cols * 4.3 / 10;
        center.height = (float) sub.rows * 4.3 / 10;

        int numInCenter = 0;

//        cout << center << endl;

        // it means the detection is not very accurate so we might have several points of intersection
        // check if most of points are located inside the center area of the image
        for (vector<Point2f>::iterator pointIt = intersections.begin();
             pointIt != intersections.end(); pointIt++) {
            Point2f p = *pointIt;

//            printf("inter x %.2f y %.2f\n", p.x, p.y);

            if (p.x > center.x && p.x < center.x + center.width && p.y > center.y &&
                p.y < center.y + center.height) {
                numInCenter++;
            }
        }

        printf("number in center %d total intersections %d percentage %.2f\n", numInCenter, intersections.size(),
               float(numInCenter) / intersections.size() * 100);

        if (float(numInCenter) / intersections.size() > LIMIT_PERCENTAGE_OF_POINTS) {
            result.insert(result.end(), *it);
        }
    }

    printf("result size %d\n", result.size());

    return result;
}

vector<Rect> concentric_intersection(const Mat &image) {
    vector<Vec3f> circles = hough_circle(image, 12, 40, 80);

    vector<Rect> det_circles;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat blank = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1);

    for (Vec3f c : circles) {
        Point center(c.val[1], c.val[0]);
        circle(blank, center, c.val[2], Scalar(255, 0, 0), -1);
    }

    // find the connected components
    findContours(blank, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // get the moments from the contours
    vector<Moments> mu(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        mu[i] = moments(contours[i], false);
    }

    // get the centroid
    vector<Point2f> centers(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        centers[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }

    // draw contours
    Mat drawing(blank.size(), CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(255, 0, 0); // B G R values
//        drawContours(image, contours, i, color, 2, 8, hierarchy, 0, Point());
//        circle(image, centers[i], 4, color, -1, 8, 0);
        det_circles.emplace_back(boundingRect(contours[i]));
    }

    return det_circles;
}

vector<Rect> template_matching(const Mat &src, vector<Rect> targets) {
    vector<Rect> result;
    // load templates
    int sizes[] = {20, 25, 30, 35, 40, 45, 50};
    vector<Mat> templates;
    for (int size : sizes) {
        char output[50];
        sprintf(output, "../templates/templ%d.png", size);
        templates.push_back(imread(output));
    }

    int match_method = CV_TM_CCOEFF_NORMED;

    for (const Rect &target : targets) {
        Range rows(target.y, target.y + target.height);
        Range cols(target.x, target.x + target.width);
        Mat sub = src(rows, cols).clone();

//        imshow("Source Image", sub);
//        waitKey(0);

        // match templates according to the size of a target
        double max = 0;
        double min = 2;
        double scoreSum = 0;
        vector<double> vals;
        for (Mat templ : templates) {
            if (target.width < templ.cols || target.height < templ.rows) {
                cout << "filtered" << endl;
                break;
            }

            // Create the result matrix
            int result_cols = sub.cols - templ.cols + 1;
            int result_rows = sub.rows - templ.rows + 1;

            Mat matchingResult(result_rows, result_cols, CV_32F);

            // Do the Matching and Normalize
            matchTemplate(sub, templ, matchingResult, match_method);

            // Localizing the best match with minMaxLoc
            double minVal;
            double maxVal;
            Point minLoc;
            Point maxLoc;
            Point matchLoc;
            minMaxLoc(matchingResult, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
            matchLoc = maxLoc;

            vals.push_back(maxVal);

            max = max > maxVal ? max : maxVal;
            min = min > maxVal ? maxVal : min;
            scoreSum += maxVal;

//            imshow("Source Image", temp);
//            waitKey(0);
        }

        sort(vals.begin(), vals.end());

        double median;

        if (vals.size() % 2 == 0) {
            median = (vals[vals.size() / 2 - 1] + vals[vals.size() / 2]) / 2;
        } else {
            median = vals[vals.size() / 2];
        }

        if (max > 0.51) {
            result.emplace_back(target);
        }

        cout << "max " << max << " min " << min << " score avg " << scoreSum / vals.size() << " median " << median
             << endl;
    }

    return result;
}


/**
 * Detect dartboards using speeded up robust features
 *
 * code adapted from:
 *
 * http://www.coldvision.io/2016/06/27/object-detection-surf-knn-flann-opencv-3-x-cuda/
 *
 * http://answers.opencv.org/question/16548/object-detection-using-surf/
 *
 * @param im_scene: input image
 * @return good points: All detected points on a dartboard
 */
vector<Rect> filter_SURF(Mat im_scene, vector<Rect> det_boards) {

    // Read in image and resize to
    Mat image = im_scene;
    Mat im_object = imread("../training_data/dart.bmp", 0);
    resize(im_object, im_object, Size(175, 175));
    cvtColor(im_scene, im_scene, COLOR_RGB2GRAY);

    vector<KeyPoint> keypoints_object, keypoints_scene; // keypoints

    Mat descriptors_object, descriptors_scene; // descriptors (features)

    int minHessian = 400;

    // Detect keypoints and decriptors
    Ptr<SURF> surf = SURF::create(minHessian);
    surf->detectAndCompute(im_object, noArray(), keypoints_object, descriptors_object);
    surf->detectAndCompute(im_scene, noArray(), keypoints_scene, descriptors_scene);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher; // FLANN - Fast Library for Approximate Nearest Neighbors
    vector<vector<DMatch> > matches;
    matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2); // find the best 2 matches of each descriptor


    //-- Step 4: Select only good matches
    std::vector<DMatch> good_matches;
    for (int k = 0; k < std::min(descriptors_scene.rows - 1, (int) matches.size()); k++) {
        if ((matches[k][0].distance < 0.95 * (matches[k][1].distance)) &&
            ((int) matches[k].size() <= 2 && (int) matches[k].size() > 0)) {
            // take the first result only if its distance is smaller than 0.6*second_best_dist
            // that means this descriptor is ignored if the second distance is bigger or of similar
            good_matches.push_back(matches[k][0]);
        }
    }

    //-- Step 5: Draw lines between the good matching points
    Mat img_matches;
    drawMatches(im_object, keypoints_object, im_scene, keypoints_scene,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::DEFAULT);

    // Get the good keypoints
    std::vector<Point2f> good_points;
    for (int i = 0; i < good_matches.size(); i++) {

        good_points.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }

    printf("Keypoints scene: %d\n",(int) keypoints_scene.size());
    printf("Good points: %d\n",(int) good_points.size());


    imshow("Good Matches & Object detection", img_matches);
    waitKey(0);


    int del = 0;
    for (int i = 0; i < det_boards.size(); i++) {

        for (Point2f point: good_points) {
            // if rect contains a point
            if (det_boards[i].contains(point)) {
                del++;
            }
        }
        if (del < 1 && det_boards.size() > 0){
            det_boards.erase(det_boards.begin()+i);
        }
        del = 0;
    }

    return det_boards;

}
