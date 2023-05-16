#include <stdio.h>
#include <map>
#include "opencv_aee.hpp"
#include "main.hpp"
#include "pi2c.h"

#define frameWidth 320
#define frameHeight 240

#define PINKLOWER Scalar(145, 68, 57)
#define PINKUPPER Scalar(170, 255, 255)

// define colour filter HSV boundaries
#define RED1LOWER Scalar(170, 68, 57)
#define RED1UPPER Scalar(179, 255, 255)
#define RED2LOWER Scalar(0, 68, 57)
#define RED2UPPER Scalar(6, 255, 255)
#define GREENLOWER Scalar(39, 68, 57)
#define GREENUPPER Scalar(78, 255, 255)
#define BLUELOWER Scalar(90, 68, 57)
#define BLUEUPPER Scalar(130, 255, 255)
#define YELLOWLOWER Scalar(20, 68, 57)
#define YELLOWUPPER Scalar(33, 255, 255)
#define BLACKLOWER Scalar(0, 0, 0)
#define BLACKUPPER Scalar(179, 77, 64)

// define number of columns in frame
#define columnsNumber 10

// define weights for weighted average
#define w1 -5
#define w2 -4
#define w3 -3
#define w4 -2
#define w5 -1
#define w6 1
#define w7 2
#define w8 3
#define w9 4
#define w10 5

// define pid coefficients
#define Kp 0.1
#define Ki 0.2
#define Kd 5

using namespace std;

Pi2c car(0x04); // configure the I2C interface to the Car as a global variable


void imageColumns(Mat filter, int columnsPixelSum[]) {
    int columnWidth = frameWidth / columnsNumber;

    for(int c = 0; c < columnsNumber; c++) {
        int sum = 0;

        for(int x = c * columnWidth; x < (c + 1) * columnWidth; x++) {
            Mat col = filter.col(x);

            for(int y = 0; y < frameHeight; y++) {
                int pixel = col.at<int>(y);

                if(pixel != 0) {
                    sum++;
                }
            }
        }

        columnsPixelSum[c] = sum;
        //cout << sum << "\t";
    }

    //cout << "\n";
}

// counts the number of white pixels per column
float weightedAverage(int columnsPixelSum[], float weights[]) {
    float weightedSum = 0;
    float total = 0;

    for(int i = 0; i < columnsNumber; i++) {
        weightedSum += columnsPixelSum[i] * weights[i];
        total += columnsPixelSum[i];
    }

    if (total == 0) {
        return 0;
    }

    float wAvg = weightedSum / total;

    return wAvg;
}

// PID control function
long pidControl(float wAvg) {
    float prevError = 0;
    float iGain = 0;

    cout << "w=" << wAvg << endl;

    float error = 0 - wAvg;

    float pGain = Kp * error;

    for(int c = 0; c < columnsNumber; c++) {
        iGain += Ki * error;
    }

    float dGain = Kd * (error - prevError);

    long u = pGain + iGain + dGain;

    prevError = error;

    //cout << "p=" << pGain << endl;
    //cout << "i=" << iGain << endl;
    //cout << "d=" << dGain << endl;
    cout << "u=" << u << endl;

    return u;
}

void sendMechanicalData(int leftMotor, int rightMotor, int steeringAngle) {
    char data[6];
    data[0] = (leftMotor >> 8) & 0xFF;
    data[1] = leftMotor & 0xFF;
    data[2] = (rightMotor >> 8) & 0xFF;
    data[3] = rightMotor & 0xFF;
    data[4] = (steeringAngle >> 8) & 0xFF;
    data[5] = steeringAngle & 0xFF;
    car.i2cWrite(data, 6);
}

void setup(void) {
    setupCamera(frameWidth, frameHeight);
}

int main(int argc, char **argv) {
    float weights[columnsNumber] = {w1, w2, w3, w4, w5, w6, w7, w8, w9, w10};
    float pidControlInput;
    setup();

    cv::namedWindow("Photo");

    while (1) {
        Mat frame;

        Mat image_REDFILTER1;
        Mat image_REDFILTER2;
        Mat image_REDFILTER;
        Mat image_GREENFILTER;
        Mat image_BLUEFILTER;
        Mat image_YELLOWFILTER;
        Mat image_BLACKFILTER;

        // map of colour filter matrices
        map<string, Mat> filtersMap;
        filtersMap.insert({"REDFILTER1", image_REDFILTER1});
        filtersMap.insert({"REDFILTER2", image_REDFILTER2});
        filtersMap.insert({"REDFILTER", image_REDFILTER});
        filtersMap.insert({"GREENFILTER", image_GREENFILTER});
        filtersMap.insert({"BLUEFILTER", image_BLUEFILTER});
        filtersMap.insert({"YELLOWFILTER", image_YELLOWFILTER});
        filtersMap.insert({"BLACKFILTER", image_BLACKFILTER});

        map<string, float> wAvgMap;

        while (frame.empty())
            frame = captureFrame();

        rotate(frame, frame, ROTATE_180);

        Mat image_hsv;
        cvtColor(frame, image_hsv, COLOR_BGR2HSV);

        // line detection filter setup
        // filter out values outside of boundaries
        inRange(image_hsv, RED1LOWER, RED1UPPER, filtersMap["REDFILTER1"]);
        inRange(image_hsv, RED2LOWER, RED2UPPER, filtersMap["REDFILTER2"]);
        inRange(image_hsv, GREENLOWER, GREENUPPER, filtersMap["GREENFILTER"]);
        inRange(image_hsv, BLUELOWER, BLUEUPPER, filtersMap["BLUEFILTER"]);
        inRange(image_hsv, YELLOWLOWER, YELLOWUPPER, filtersMap["YELLOWFILTER"]);
        inRange(image_hsv, BLACKLOWER, BLACKUPPER, filtersMap["BLACKFILTER"]);

        filtersMap["REDFILTER"] = filtersMap["REDFILTER1"] | filtersMap["REDFILTER2"]; // combines matrices "REDFILTER1" and "REDFILTER2" into 1 matrix

        for (auto const & [filter, matrix] : filtersMap) {
            int columnNumber[columnsNumber];

            imageColumns(matrix, columnNumber);
            wAvgMap.insert({filter, weightedAverage(columnNumber, weights)});
        }

        Mat symbol_check;
        inRange(image_hsv, PINKLOWER, PINKUPPER, symbol_check);

        // find the contours of the image saving them as img_symbol_contours
        std::vector<std::vector<Point>> img_symbol_contours;
        findContours(symbol_check, img_symbol_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // draw the contours
        drawContours(frame, img_symbol_contours, -1, Scalar(0, 0, 255), 2);

        // transform the perspective of the image
        std::vector<Point2f> corners;
        for (int i = 0; i < img_symbol_contours.size(); i++) {
            if (img_symbol_contours[i].size() > 4) {
                approxPolyDP(img_symbol_contours[i], corners, arcLength(img_symbol_contours[i], true) * 0.02, true);
                if (corners.size() == 4) {
                    // draw the contours
                    drawContours(frame, img_symbol_contours, i, Scalar(0, 255, 0), 2);
                    // draw the corners
                    for (int j = 0; j < corners.size(); j++) {
                        circle(frame, corners[j], 5, Scalar(0, 0, 255), 2);
                    }
                }
            }
        }

        if (corners.size() == 4) {
            // change the perspective of symbol_check so that the 4 corners found are the new corners
            Mat perspective_transform = getPerspectiveTransform(corners, std::vector<Point2f>{Point2f(0, 0), Point2f(0, 100), Point2f(100, 100), Point2f(100, 0)});
            warpPerspective(symbol_check, symbol_check, perspective_transform, Size(100, 100));
            // scale up the image to 350x350
            resize(symbol_check, symbol_check, Size(350, 350));
            // find the contours of the image saving them as image_symbol_contours
            std::vector<std::vector<Point>> image_symbol_contours;
            findContours(symbol_check, image_symbol_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            // draw the contours
            drawContours(symbol_check, image_symbol_contours, -1, Scalar(255, 255, 255), 2);

            // make a 13 item array of Mat objects to store the symbols
            Mat symbols[13];
            // symbols[0] = imread("Circle (Red Line).png");
            symbols[0] = imread("Circle_Red_Line.png");
            // symbols[1] = imread("Star (Green Line).png");
            symbols[1] = imread("Star_Green_Line.png");
            // symbols 1-4 are the same image but roated 90 degrees
            rotate(symbols[1], symbols[2], ROTATE_90_CLOCKWISE);
            rotate(symbols[1], symbols[3], ROTATE_180);
            rotate(symbols[1], symbols[4], ROTATE_90_COUNTERCLOCKWISE);
            // symbols[5] = imread("Triangle (Blue Line).png");
            symbols[5] = imread("Triangle_Blue_Line.png");
            rotate(symbols[5], symbols[6], ROTATE_90_CLOCKWISE);
            rotate(symbols[5], symbols[7], ROTATE_180);
            rotate(symbols[5], symbols[8], ROTATE_90_COUNTERCLOCKWISE);
            // symbols[9] = imread("Umbrella (Yellow Line).png");
            symbols[9] = imread("Umbrella_Yellow_Line.png");
            rotate(symbols[9], symbols[10], ROTATE_90_CLOCKWISE);
            rotate(symbols[9], symbols[11], ROTATE_180);
            rotate(symbols[9], symbols[12], ROTATE_90_COUNTERCLOCKWISE);
            String symbol_names[13] = {"Circle", "Star", "Star", "Star", "Star", "Triangle", "Triangle", "Triangle", "Triangle", "Umbrella", "Umbrella", "Umbrella", "Umbrella"};

            int similarity[13];

            for (int i = 0; i < 13; i++) {
                // convert the symbols to hsv and black and white
                cvtColor(symbols[i], symbols[i], COLOR_BGR2HSV);
                inRange(symbols[i], PINKLOWER, PINKUPPER, symbols[i]);

                // find the contours of the symbols
                std::vector<std::vector<Point>> symbol_contours;
                findContours(symbols[i], symbol_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                // draw the contours
                drawContours(symbols[i], symbol_contours, -1, Scalar(255, 255, 255), 2);

                // use compare() to compare symbols[i] and symbol_check
                Mat result;
                compare(symbol_check, symbols[i], result, CMP_EQ);
                // store the number of pixels that are the same in a list
                int pixel_count = countNonZero(result);
                // convert the pixel count to a percentage
                pixel_count = (pixel_count / 350.0) * 100;
                similarity[i] = pixel_count;
            }
            // image similarity counter
            int max = 0;
            int max_index = 0;
            for (int i = 0; i < 13; i++) {
                if (similarity[i] > max) {
                    max = similarity[i];
                    max_index = i;
                }
            }
            // check if the percentage is above 75%
            if (max > 75) {
                // display the name of the symbol that is most similar on the frame
                putText(frame, symbol_names[max_index], Point(100, 225), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 1);

                String filter_name;

                if(symbol_names[max_index] == "Triangle"){
                    filter_name = "BLUEFILTER";
                }else if(symbol_names[max_index] == "Star"){
                    filter_name = "GREENFILTER";
                }else if(symbol_names[max_index] == "Circle"){
                    filter_name = "REDFILTER";
                }else{
                    filter_name = "YELLOWFILTER";
                }

                //cout << wAvgMap[filter_name] << endl;
                pidControlInput = wAvgMap[filter_name];

            } else {
                // display "Unknown" on the frame
                putText(frame, "Unknown", Point(100, 225), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 1);
                //cout << wAvgMap["BLACKFILTER"] << endl;
                pidControlInput = wAvgMap["BLACKFILTER"];
            }
        } else {
            // display "Unknown" on the frame
            putText(frame, "Unknown", Point(100, 225), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 1);
            //cout << wAvgMap["BLACKFILTER"] << endl;
            pidControlInput = wAvgMap["BLACKFILTER"];
        }

        imshow("POV with Overlay", frame);  // Display the image in the window
        imshow("HSV", image_hsv);
        imshow("Mask", symbol_check);

        //cout << pidControlInput << endl;
        //cout << pidControl(pidControlInput) << endl;

        int servoAngle = 90 * -(pidControl(pidControlInput));
        int motorSpeed = 110 - (0.1 * (pidControl(pidControlInput)));
        int motorSpeed2 = 110 + (0.1 * (pidControl(pidControlInput)));

        if (servoAngle > 150) {
            servoAngle = 150;
        }
        if (servoAngle < 30) {
            servoAngle = 30;
        }
        if (motorSpeed > 255) {
            motorSpeed = 255;
        }
        if (motorSpeed < -255) {
            motorSpeed = -255;
        }

        sendMechanicalData(motorSpeed, motorSpeed2, servoAngle);
        cout << "servoAngle=" << servoAngle << endl;
        cout << "motorSpeed=" << motorSpeed << endl;

        int key = cv::waitKey(1);  // Wait 1ms for a keypress (required to update windows)

        key = (key == 255) ? -1 : key;  // Check if the ESC key has been pressed
        if (key == 27)
            break;
    }

    closeCV();  // Disable the camera and close any windows

    return 0;
}
