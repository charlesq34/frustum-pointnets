/* 
mex gpc.c cuboidIntersectionVolume.c -O -output cuboidIntersectionVolume              % optimized
mex gpc.c cuboidIntersectionVolume.c -argcheck -output cuboidIntersectionVolume       % with argument checking
mex gpc.c cuboidIntersectionVolume.c -g -output cuboidIntersectionVolume              % for debugging 
*/

#include "mex.h"
#include "gpc.h"
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

/* ===============================
	    Constants
	===============================*/

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/*	=================================
		GATEWAY ROUTINE TO MATLAB
	=================================*/

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    unsigned int i,j,n1,n2,c,v,m;
    double* volume;
    double* b1;
    double* b2;
    unsigned int joffset;
    double zOverlap;
    double areaOverlap;
    double* result_vertex;
    gpc_polygon subject, clip, result;
    gpc_vertex_list subject_contour;
    gpc_vertex_list clip_contour;
    int hole = 0;
    
    subject.num_contours = 1;
    subject.hole = &hole;
    subject.contour = &subject_contour;
    subject.contour[0].num_vertices = 4;
    
    clip.num_contours = 1;
    clip.hole = &hole;
    clip.contour = &clip_contour;
    clip.contour[0].num_vertices = 4;
    
    n2 = mxGetN(prhs[0]);
    n1 = mxGetN(prhs[1]);
    b2 = mxGetPr(prhs[0]);
    b1 = mxGetPr(prhs[1]);

    plhs[0] = mxCreateNumericMatrix(n2, n1, mxDOUBLE_CLASS, mxREAL);
    volume = (double*) mxGetData(plhs[0]);
    
    for (i=0; i<n1; ++i){
        /* subject */
        subject.contour[0].vertex = (gpc_vertex *)b1;
        for (j=0; j<n2; ++j){
            joffset = 10*j;
            zOverlap = MIN(b1[9],b2[joffset+9]) - MAX(b1[8],b2[joffset+8]);

            if (zOverlap>0){
                /* get intersection */ 
                clip.contour[0].vertex = (gpc_vertex *)(b2+joffset);
                   
                gpc_polygon_clip(1, &subject, &clip, &result);
                              
                
                if (result.num_contours>0 && result.contour[0].num_vertices > 2) { 
                    /* compute area of intersection */

                    /*
                     * http://www.mathopenref.com/coordpolygonarea.html
                     * Green's theorem for the functions -y and x; 
                     http://stackoverflow.com/questions/451426/how-do-i-calculate-the-surface-area-of-a-2d-polygon
                     */
                    result_vertex = (double*)(result.contour[0].vertex);
                    m = result.contour[0].num_vertices;
                    areaOverlap = (result_vertex[2*m-2]*result_vertex[1]-result_vertex[2*m-1]*result_vertex[0]);
                    for (v= 1; v < m; v++)
                    {
                        areaOverlap += (result_vertex[v*2-2]*result_vertex[v*2+1]-result_vertex[v*2-1]*result_vertex[v*2]);
                    }
                    *volume = zOverlap * 0.5 * fabs(areaOverlap);
                    
                    
                }
                gpc_free_polygon(&result);
            }            
            ++volume;
        }
        b1+=10;
    }
    /*
    gpc_free_polygon(&subject);
    gpc_free_polygon(&clip);
    
    mxFree(subject.hole);
    mxFree(subject.contour);
    mxFree(clip.hole);
    mxFree(clip.contour);   
     **/ 
    
}
