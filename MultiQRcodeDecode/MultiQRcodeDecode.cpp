#include "stdafx.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <zbar.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

using namespace std;
using namespace cv;
using namespace zbar;

int thresh = 500;
IplImage* img = 0;
IplImage* img0 = 0;
CvMemStorage* storage = 0;
CvPoint pt[4], pt_sorted[4];
const char* wndname = "Square Detection";

//finds a cosine of angle between vectors
//from pt0->pt1 and from pt0->pt2
double angle(CvPoint* pt1, CvPoint* pt2, CvPoint* pt0)
{   
    double dx1 = pt1->x - pt0->x;   
    double dy1 = pt1->y - pt0->y;   
    double dx2 = pt2->x - pt0->x;
    double dy2 = pt2->y - pt0->y;

    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2));
}
 
//returns sequence of squares detected on the image.
//the sequence is stored in the specified memory storage
CvSeq* findSquares4(IplImage* img, CvMemStorage* storage)
{
    CvSeq* contours;
    int i, c, l, N = 11;
    CvSize sz = cvSize( img->width & -2, img->height & -2 );
    IplImage* timg = cvCloneImage( img ); // make a copy of input image
    IplImage* gray = cvCreateImage( sz, 8, 1 );
    IplImage* pyr = cvCreateImage( cvSize(sz.width/2, sz.height/2), 8, 3 );
    IplImage* tgray;

    CvSeq* result;
    double s, t;

    //create empty sequence that will contain points -
    //4 points per square (the square's vertices)
    CvSeq* squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );

    //select the maximum ROI in the image
    //with the width and height divisible by 2
    cvSetImageROI( timg, cvRect( 0, 0, sz.width, sz.height )); 

    //down-scale and upscale the image to filter out the noise
    cvPyrDown( timg, pyr, 7 );
    cvPyrUp( pyr, timg, 7 );
    tgray = cvCreateImage( sz, 8, 1 );    

    //find squares in every color plane of the image
    for( c = 0; c < 3; c++ )
    {    
        //extract the c-th color plane
        cvSetImageCOI( timg, c+1 );
        cvCopy( timg, tgray, 0 ); 
        
    	//try several threshold levels   
     	for( l = 0; l < N; l++ )    
    	{        
        	//hack: use Canny instead of zero threshold level.      
       		//Canny helps to catch squares with gradient shading      
      		if( l == 0 )      
      		{
			     
            	//apply Canny. Take the upper threshold from slider 
            	//and set the lower to 0 (which forces edges merging)
            	cvCanny( tgray, gray, 0, thresh, 5 );  
                
            	//dilate canny output to remove potential   
            	//holes between edge segments 
            	cvDilate( gray, gray, 0, 1 );        
            	cvShowImage("gray", gray); 
        	}    

        else  
        {      

    		//apply threshold if l!=0:
            //tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0 
            cvThreshold( tgray, gray, (l+1)*255/N, 255, CV_THRESH_BINARY ); 
        } 

        //find contours and store them all as a list
        cvFindContours( gray, storage, &contours, sizeof(CvContour),  
        CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );  

        //test each contour
        	while( contours )
            {
            	//approximate contour with accuracy proportional
                //to the contour perimeter
                result = cvApproxPoly( contours, sizeof(CvContour), storage,
                CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );
                
                //square contours should have 4 vertices after approximation
                //relatively large area (to filter out noisy contours)
                //and be convex.
                //Note: absolute value of an area is used because
                //area may be positive or negative - in accordance with the
                //contour orientation
                
                if( result->total == 4 &&
                    fabs(cvContourArea(result,CV_WHOLE_SEQ)) > 1000 &&
                    cvCheckContourConvexity(result) )
                {
                    s = 0;
                    for( i = 0; i < 5; i++ )
                    {

                    	//find minimum angle between joint 
						//edges (maximum of cosine)
                        if( i >= 2 )

                        {
                            t = fabs(angle(
                            (CvPoint*)cvGetSeqElem( result, i ),
                            (CvPoint*)cvGetSeqElem( result, i-2 ),
                            (CvPoint*)cvGetSeqElem( result, i-1 )));
                            s = s > t ? s : t;
                        }
                    }
            
                //if cosines of all angles are small 
                //(all angles are ~90 degree) then write quandrange
                //vertices to resultant sequence
                	if( s < 0.3 )
                        for( i = 0; i < 4; i++ ) 
                           cvSeqPush( squares, 
                               (CvPoint*)cvGetSeqElem( result, i ));
                }
                //take the next contour 
            	contours = contours->h_next;
            }
        }
    }
    //release all the temporary images
    cvReleaseImage( &gray );
    cvReleaseImage( &pyr );
    cvReleaseImage( &tgray );
    cvReleaseImage( &timg );
    
    return squares;
}

//the function draws all the squares in the image
void drawSquares( IplImage* img, CvSeq* squares )
{   
    CvSeqReader reader;
    IplImage* cpy = cvCloneImage( img );
    int i;

    //initialize reader of the sequence
    cvStartReadSeq( squares, &reader, 0 );
    
    //read 4 sequence elements at a time (all vertices of a square)
    for( i = 0; i < squares->total; i += 4 )
    {
        CvPoint* rect = pt;
        int count = 4;
 
        //read 4 vertices
        memcpy( pt, reader.ptr, squares->elem_size );
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );
        
        memcpy( pt + 1, reader.ptr, squares->elem_size );
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );

        memcpy( pt + 2, reader.ptr, squares->elem_size );
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );

        memcpy( pt + 3, reader.ptr, squares->elem_size );
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );

        //draw the square as a closed polyline
        cvPolyLine( cpy, &rect, &count, 1, 1, CV_RGB(0,0,255), 2, CV_AA, 0 );
    }

	//show the resultant image
	cvShowImage( wndname, cpy );
	cvReleaseImage( &cpy );
}

//sort points in order to warp the perdpective properly 
//with sequence of up-left, up-right, down-right, down-left
void PointSort()
{
	int Sum = 0, Sub = 0;

	//the point with smallest value(x plus y)  will be the up-left point
	Sum = pt[0].x + pt[0].y;
	pt_sorted[0] = pt[0];
	for(int j=1; j<4; j++)
	{
		if(pt[j].x + pt[j].y < Sum)
		{
			Sum = pt[j].x + pt[j].y;
			pt_sorted[0] = pt[j];
		}
	}

	//the point with the largest value(x minus y) will be the up-right point
	Sub = pt[0].y - pt[0].x;
	pt_sorted[1] = pt[0]; 
	for(int j=1; j<4; j++)
	{
		if(pt[j].y - pt[j].x < Sub)
		{
			Sub = pt[j].y - pt[j].x;
			pt_sorted[1] = pt[j];
		}
	}

	//the point with the largest value(x plus y) will be the down-right point
	Sum = pt[0].x + pt[0].y;
	pt_sorted[2] = pt[0];
	for(int j=1; j<4; j++)
	{
		if(pt[j].x + pt[j].y > Sum)
		{
			Sum = pt[j].x + pt[j].y;
			pt_sorted[2] = pt[j];
		}
	}
	
	//the point with the smallest value(x minus y) will be the down-left point
	Sub = pt[0].y - pt[0].x;
	pt_sorted[3] = pt[0]; 
	for(int j=1; j<4; j++)
	{
		if(pt[j].y - pt[j].x > Sub)
		{
			Sub = pt[j].y - pt[j].x;
			pt_sorted[3] = pt[j];
		}
	}		
}

//warp the perspective in order to crop the image that contains QRcode only
IplImage* WarpPerspective(IplImage* input)
{
	IplImage* output;
	output = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 3);

	CvMat* mmat = cvCreateMat(3,3,CV_32FC1);
    
	CvPoint2D32f *c1 = new CvPoint2D32f[4];
	CvPoint2D32f *c2 = new CvPoint2D32f[4];
	
	c1[0].x = pt_sorted[0].x;	c1[0].y = pt_sorted[0].y;
	c1[1].x = pt_sorted[1].x;	c1[1].y = pt_sorted[1].y;
	c1[2].x = pt_sorted[2].x;	c1[2].y = pt_sorted[2].y;
	c1[3].x = pt_sorted[3].x;	c1[3].y = pt_sorted[3].y;
	
	c2[0].x = 10;	c2[0].y = 10;
	c2[1].x = 100;	c2[1].y = 10;
	c2[2].x = 100;	c2[2].y = 100;
	c2[3].x = 10;	c2[3].y = 100;
    
	mmat = cvGetPerspectiveTransform(c1, c2, mmat);
	cvWarpPerspective(input, output, mmat);

	cvShowImage("Original", input);
	cvShowImage("Warp", output);

	return output;
}

//crop the image that only contains QRcode
IplImage* imageROI(IplImage* warp_img)
{
	IplImage* warp_img_ROI;

	CvRect rect_ROI = cvRect(10, 10 , 90, 90);

	warp_img_ROI = cvCloneImage(warp_img);
	cvSetImageROI(warp_img_ROI, rect_ROI);

	cvShowImage("ROI", warp_img_ROI);

	return warp_img_ROI;
}

//decode for QRcode 
void QRcode_Decode(IplImage* warp_img_ROI)
{
	ImageScanner scanner;    
	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);   
    
	Mat img_ROI = cvarrToMat(warp_img_ROI);  
	
	Mat imgout;
	cvtColor(img_ROI,img_ROI,CV_RGB2GRAY);
	
	imgout = img_ROI;
    
	string qrcode_data;
	int width = img_ROI.cols;    
	int height = img_ROI.rows;    
	uchar *raw = (uchar *)img_ROI.data;       
	Image image(width, height, "Y800", raw, width * height);      
	int n = scanner.scan(image);      
	for(Image::SymbolIterator symbol = image.symbol_begin();symbol != image.symbol_end();++symbol)  
	{    
		vector<Point> vp;    
		cout<<"Decoded¡G"<<endl<<symbol->get_type_name()<<endl<<endl;  
		cout<<"Symbol¡G"<<endl<<symbol->get_data()<<endl<<endl;           
		int n = symbol->get_location_size();    
		for(int i=0;i<n;i++)  
		{    
			vp.push_back(Point(symbol->get_location_x(i),symbol->get_location_y(i)));   
		}    
		RotatedRect r = minAreaRect(vp);    
		Point2f pts[4];    
		r.points(pts);    
		Point textPoint(pts[1]);  
		qrcode_data=symbol->get_data();  
      
		cout<<"Angle: "<<r.angle<<endl;   
		//put the decoded text at the up left corner of the QRcode
		putText(cvarrToMat(img),qrcode_data,pt_sorted[0],FONT_HERSHEY_COMPLEX,1,Scalar(0,0,255),1,8,false);
    }    
	zbar_image_free_data(image);
	img_ROI.release();
	imgout.release();
}

//set the threshold for the cvcanny fuction
void on_trackbar( int a )
{
	if( img )
    	drawSquares( img, findSquares4( img, storage ) );
}

int main(int argc, char** argv)
{
	int c = 0;

	// create memory storage that will contain all the dynamic data
	storage = cvCreateMemStorage(0);

	CvCapture* cap = cvCreateCameraCapture(0);
	img0 = cvQueryFrame(cap);
	img = cvCloneImage( img0 );
	IplImage* warp_img;
	IplImage* warp_img_ROI;

	// create window and a trackbar with parent 
	cvNamedWindow( wndname, 1 );
	cvCreateTrackbar( "canny thresh", wndname, &thresh, 1000, on_trackbar );

	// force the image processing
	on_trackbar(0);

	while(true)
	{
		storage = cvCreateMemStorage(0);

		CvCapture* cap = cvCreateCameraCapture(0);

		img0 = cvQueryFrame(cap);
		img = cvCloneImage(img0);
		IplImage* warp_img;
		IplImage* warp_img_ROI;

		if( img )
		{
			drawSquares( img, findSquares4( img, storage ) );
			PointSort();
			warp_img = WarpPerspective(img);
			warp_img_ROI = imageROI(warp_img);
			QRcode_Decode(warp_img_ROI);
		}
		cvWaitKey(1);
		
		cvReleaseImage(&img);
		cvReleaseImage(&warp_img);
		cvReleaseImage(&warp_img_ROI);
		cvReleaseImage(&img0);

		cvReleaseMemStorage( &storage ); 		

		char c = waitKey(1);
		if(c == 27)
			break;
	}
	return 0;
}
