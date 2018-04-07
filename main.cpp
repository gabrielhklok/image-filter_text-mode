#include <stdio.h>
#include <iostream>
#include <string>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "filter.hpp"

using namespace std;
using namespace cv;

void webcam(Filter*);
void photo(Filter*, string, string, int);
void help();
Mat choiceFilter(Filter*, Mat, int);
void choiceKey(int, int&, bool&);

int main(int argc, char** argv){
  string parameter = argv[1];

  Filter *filter = new Filter();
  
  if(parameter.compare("-webcam") == 0){

    webcam(filter);

  }else if(parameter.compare("-photo") == 0){

    photo(filter, argv[2], argv[3], atoi(argv[4]));

  }else if(parameter.compare("--help") == 0){
    help();
  }else{
    cout << "Not found parameter. Please use command '--help' for details." << endl;
  }

  return(1);
}


/**
 *
 */
void webcam(Filter *filter){
  int key = 0;
  bool loop = true;
  VideoCapture cap(0);
   
  if (!cap.isOpened()) {
    cout << "Can't open camera!" << endl;
  }

  while(loop){
    Mat frame;

    cap >> frame;
    if (frame.empty())
      break;    
    
    imshow( "Webcam", choiceFilter(filter, frame, key) );
    
    choiceKey(waitKey(1), key, loop);
  }
}

/**
 *
 */
void photo(Filter *filter, string urlImgIn, string urlImgOut, int key){
  Mat imgIn, imgOut;

  imgIn = imread(urlImgIn);
  if(!imgIn.data){
    cout << "Can't open to image: " + urlImgIn << endl;
  }else{

    //filter->negative(imgIn, imgOut);
    imshow("Photo", choiceFilter(filter, imgIn, key));
    waitKey(0);
  }
}

/**
 *
 */
Mat choiceFilter(Filter *filter, Mat frame, int key){
  Mat frameResponse, frameHistogram;

  switch(key){
    case 1:
      filter->grayscale(frame, frameResponse);
      filter->histogram(frameResponse, frameHistogram);
      break;

    case 2:
      filter->grayscale(frame, frameResponse);
      filter->histogram(frameResponse, frameHistogram);
      break;

    case 3:
      filter->thresholding(frame, 230, frameResponse);
      filter->histogram(frameResponse, frameHistogram);
      break;

    case 4:
      filter->negative(frame, frameResponse);
      filter->histogram(frameResponse, frameHistogram);
      break;

    case 5:
      filter->zoomIn(frame, frameResponse);
      filter->histogram(frameResponse, frameHistogram);
      break;

    case 6:
      filter->zoomOut(frame, frameResponse);
      filter->histogram(frameResponse, frameHistogram);
      break;

    case 7:
      filter->thresholding(frame, 230, frameResponse);
      filter->outgoingPoints(frameResponse, frameResponse);
      filter->histogram(frameResponse, frameHistogram);
      break;

    default:
      frameResponse = frame;
      filter->histogram(frameResponse, frameHistogram);
  }

  return frameResponse;
}

/**
 *
 */
void choiceKey(int value, int &key, bool &loop){
  switch(value){
    case 27:
      loop = false;
      break;

    case 48:
      key = 0;
      break;

    case 49:
      key = 1;
      break;

    case 50:
      key = 2;
      break;
    
    case 51:
      key = 3;
      break;

    case 52:
      key = 4;
      break;
  
    case 53:
      key = 5;
      break;

    case 54:
      key = 6;
      break;

    case 55:
      key = 7;
      break;
  }
}

/**
 *
 */
void help(){
  cout << "Ajuda:" << endl;
  cout << endl;
  cout << "./main [OPTION] [PARAM1] [PARAM2]" << endl;
  cout << endl;
  cout << endl;
  cout << "./main -webcam         utiliza webcam" << endl;
  cout << "./main -photo [URL-IMAGE-IN] [URL-IMAGE-OUT]" << endl;
}

