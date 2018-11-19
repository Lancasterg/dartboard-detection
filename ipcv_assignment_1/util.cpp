//
// Created by Jeaung on 11/16/18.
//

#include "header/hough.h"
#include "header/util.h"

vector<Rect> line_intersection(const Mat &src, vector<Rect> &circles) {
    printf("original rows %d cols %d\n", src.rows, src.cols);

    vector<Rect> result;

    for (vector<Rect>::iterator it = circles.begin(); it != circles.end(); it++) {
        printf("detected rectangles x %d  y %d width %d height %d\n", it->x, it->y, it->width, it->height);

        Range rows(it->y, it->y + it->height);
        Range cols(it->x, it->x + it->width);

        Mat sub = src(rows, cols).clone();

//        imshow("aaa", sub);
//        waitKey(0);

        // detect lines in a sub-region
        // TODO param adaptive
        vector<Vec2f> lines = hough_line(sub, it->width / 2.5, 15);

        if (lines.size() < MIN_LINES) { // only one line detected, it's not a dartboard
            continue;
        }

        // find all points of intersection in a sub-region
        vector<Point2f> intersections;

        for (vector<Vec2f>::iterator lineIt1 = lines.begin(); lineIt1 != lines.end(); lineIt1++) {
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

            for (vector<Vec2f>::iterator lineIt2 = lines.begin(); lineIt2 != lines.end(); lineIt2++) {
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
        center.x = (float) sub.cols * 3 / 10;
        center.y = (float) sub.rows * 3 / 10;
        center.width = (float) sub.cols * 4 / 10;
        center.height = (float) sub.rows * 4 / 10;

        int numInCenter = 0;

        // it means the detection is not very accurate so we might have several points of intersection
        // check if most of points are located inside the center area of the image
        for (vector<Point2f>::iterator pointIt = intersections.begin();
             pointIt != intersections.end(); pointIt++) {
             Point2f p = *pointIt;

            printf("inter x %.2f y %.2f\n", p.x, p.y);

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