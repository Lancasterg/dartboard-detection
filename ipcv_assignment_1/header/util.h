//
// Created by Jeaung on 11/18/18.
//

#ifndef COMS30121_ASSIGNMENT_UTIL_H

#define LIMIT_PERCENTAGE_OF_POINTS .19
#define MIN_LINES 1

vector<Rect> line_intersection(const Mat &src, vector<Rect> &circles);
vector<Rect> concentric_intersection(const Mat &src);
vector<Rect> sliding_window_classification(Mat &src, CascadeClassifier model);
vector<Rect> template_matching(const Mat &src, vector<Rect> targets);


#define COMS30121_ASSIGNMENT_UTIL_H

#endif //COMS30121_ASSIGNMENT_UTIL_H
