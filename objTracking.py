'''
    File name         : objTracking.py
    Description       : Main file for object tracking
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Python Version    : 3.7
'''

import cv2
from Detector import detect
from KalmanFilter import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import math

def main(fileName='video_randomball.avi', missing_handled='no', u_x=1, u_y=1, Gamma=np.array([[100,100],[100,100]]), Sigma=np.array([[100,100],[100,100]])):

    # Create opencv video capture object
    VideoCap = cv2.VideoCapture(fileName)
    total = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))

    #Variable used to control the speed of reading the video
    ControlSpeedVar = 100  #Lowest: 1 - Highest:100

    HiSpeed = 100

    #Create KalmanFilter object KF
    #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    KF = KalmanFilter(0.1, u_x, u_y, 1, 0.1,0.1)

    sd_y=math.sqrt(KF.R[1,1])
    

    debugMode=1
    
    w=np.transpose(np.random.multivariate_normal([0,0],Gamma,total))
    v=np.transpose(np.random.multivariate_normal([0,0],Sigma,total))
    
    rmse_e=[]
    rmse_p=[]
    
    measured=np.empty((2,total))
    contaminated=np.empty((2,total))
    predicted=np.empty((2,total))
    filtered=np.empty((2,total))

    measured[:]=np.nan
    contaminated[:]=np.nan
    predicted[:]=np.nan
    filtered[:]=np.nan

    matrixIndex=0
    while(True):
        # Read frame
        ret, frame = VideoCap.read()

        if ret==False:
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        # Detect object
        centers_true = detect(frame,debugMode)
        centers=[]

        for i in centers_true:
            centers.append(np.array([i[0]+w[0,matrixIndex],i[1]+v[0,matrixIndex]]))
        
        # If centroids are detected then track them
        if (len(centers_true) > 0):

            # Draw the detected circlemΩx   a
            cv2.circle(frame, (int(centers_true[0][0]), int(centers_true[0][1])), 10, (0, 191, 255), 2)

            # Predict
            (x, y) = KF.predict()
            # Draw a rectangle as the predicted object position

            cv2.rectangle(frame, (int(x - 30), int(y - 30)),(int( x + 30), int(y + 30)), (255, 0, 0), 2)
            # Update
            (x1, y1) = KF.update(centers[0])

            # Draw a rectangle as the estimated object position
            cv2.rectangle(frame, (int(x1 - 30), int(y1 - 30)), (int(x1 + 30), int(y1 + 30)), (0, 0, 255), 2)

            cv2.putText(frame, "Filtered Position", (int(x1 + 30), int(y1 +25)), 0, 0.5, (0, 0, 255), 2)
            
            filtered[0,matrixIndex]=x1.item(0)
            filtered[1,matrixIndex]=y1.item(0)
           
            cv2.putText(frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)
            
            predicted[0,matrixIndex]=x.item(0)
            predicted[1,matrixIndex]=y.item(0)
            
            cv2.putText(frame, "Measured Position", (int(centers_true[0][0] + 15), int(centers_true[0][1] - 15)), 0, 0.5, (0,191,255), 2)

            measured[0,matrixIndex]=centers_true[0][0][0]
            measured[1,matrixIndex]=centers_true[0][1][0]

            contaminated[0,matrixIndex]=centers[0][0][0]
            contaminated[1,matrixIndex]=centers[0][1][0]
            
        elif missing_handled=='yes':
            (x, y) = KF.predict()
            # Draw a rectangle as the predicted object position
            cv2.rectangle(frame, (int(x - 30), int(y - 30)),(int( x + 30), int(y + 30)), (255, 0, 0), 2)

            (x1, y1) = KF.updateMissing()

            # Draw a rectangle as the estimated object position
            cv2.rectangle(frame, (int(x1 - 30), int(y1 - 30)), (int(x1 + 30), int(y1 + 30)), (0, 0, 255), 2)

            cv2.putText(frame, "Filtered Position", (int(x1 + 30), int(y1 +25)), 0, 0.5, (0, 0, 255), 2)
            
            filtered[0,matrixIndex]=x1.item(0)
            filtered[1,matrixIndex]=y1.item(0)
           
            cv2.putText(frame, "Predicted Position", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)
            
            predicted[0,matrixIndex]=x.item(0)
            predicted[1,matrixIndex]=y.item(0)
            
        elif missing_handled=='no':
            filtered[0,matrixIndex]=np.nan
            filtered[1,matrixIndex]=np.nan
            predicted[0,matrixIndex]=np.nan
            predicted[1,matrixIndex]=np.nan

            

        matrixIndex+=1
        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            print(predicted.shape)
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed-ControlSpeedVar+1)
        
    return (measured, contaminated, predicted, filtered, total, sd_y)

def rmse(x, x2):
    tot=0
    for i in range(len(x)):
        tot+=(x[i]-x2[i])

    return math.sqrt(abs(tot)/len(x))


def differentNoises():
    noises=[np.array([[100,100],[100,100]]), np.array([[200,200],[200,200]]),  np.array([[300,300],[300,300]]),  np.array([[400,400],[400,400]]),  np.array([[500,500],[500,500]]), np.array([[600,600],[600,600]]), np.array([[1000,1000],[1000,1000]])]
    
    rmse_x=[]
    rmse_y=[]
    
    for i in range(7):
        measured, contaminated, predicted, filtered, total, sd_y=main( Gamma=noises[i], Sigma=noises[i])
        
        predicted = predicted[np.logical_not(np.isnan(predicted))].reshape(2,-1)
        filtered = filtered[np.logical_not(np.isnan(filtered))].reshape(2,-1)
        contaminated = contaminated[np.logical_not(np.isnan(contaminated))].reshape(2,-1)
        measured = measured[np.logical_not(np.isnan(measured))].reshape(2,-1)

        t=np.arange(total-1)
        plt.xlabel('time')
        plt.ylabel('Position x')
        plt.plot(t, contaminated[0,:], color='r', label='contaminated', linewidth=0.4)
        plt.plot(t, measured[0,:] , label='measured', linewidth=0.4)
        plt.fill_between(t, sd_y, alpha=0.2)
        #plt.errorbar(t, measured[0,:], yerr=sd_y, label='measured', linewidth=0.4, ecolor='black')
        plt.plot(t, predicted[0,:], color='black', label='predicted', linewidth=0.4)
        plt.plot(t, filtered[0,:] , color='g', label='filtered', linewidth=0.4)
        plt.legend()
        plt.show()

        rmse_f=rmse(measured[0,:], filtered[0,:len(measured[0,:])])
        rmse_p=rmse(measured[0,:], predicted[0,:len(measured[0,:])])

        print("measured vs predicted RMSE of x when noise= ",noises[i][0][0],": ",rmse_f)

        rmse_x.append(rmse_f)

        rmse_f=rmse(measured[1,:], filtered[1,:len(measured[0,:])])

        rmse_y.append(rmse_f)

        print("measured vs filtered RMSE  of y when noise= ",noises[i][0][0],": ",rmse_f)

        noises[i]=math.sqrt(noises[i][0][0])

    plt.suptitle("RMSE vs noise")
    plt.xlabel("noise")
    plt.ylabel("RMSE")
    plt.scatter(noises, rmse_x, label="x")
    plt.scatter(noises, rmse_y, label="y")
    plt.legend()
    plt.show()


def differentAcc():
    measured, contaminated, predicted, filtered, total, sd_y=main()
    measured, contaminated1, predicted1, filtered1, total, sd_y=main(u_x=200,u_y=200)
    measured, contaminated2, predicted2, filtered2, total, sd_y=main(u_x=200,u_y=100)

    predicted=[predicted, predicted1, predicted2]
    filtered=[filtered, filtered1, filtered2]
    contaminated=[contaminated, contaminated1, contaminated2]
    titles=["a=(1,1)", "a=(200,200)", "a=(200,100)"]

    measured = measured[np.logical_not(np.isnan(measured))].reshape(2,-1)
    predicted1 = predicted1[np.logical_not(np.isnan(predicted1))].reshape(2,-1)
    predicted2 = predicted2[np.logical_not(np.isnan(predicted2))].reshape(2,-1)
    filtered1 = filtered1[np.logical_not(np.isnan(filtered1))].reshape(2,-1)
    filtered2 = filtered2[np.logical_not(np.isnan(filtered2))].reshape(2,-1)
    contaminated1 = contaminated1[np.logical_not(np.isnan(contaminated1))].reshape(2,-1)
    contaminated2 = contaminated2[np.logical_not(np.isnan(contaminated2))].reshape(2,-1)

    for i in range(3):
        plt.xlabel('t')
        plt.ylabel('Position x')
        plt.suptitle(titles[i])
        t=np.arange(len(contaminated[i][0,:]))
        plt.plot(t, contaminated[i][0,:], 'r', label='contaminated')
        t=np.arange(measured.shape[1])
        plt.plot(t, measured[0,:], label='measured')
        plt.fill_between(t, sd_y, alpha=0.2)
        t=np.arange(len(predicted[i][0,:]))
        plt.plot(t, predicted[i][0,:], 'b', label='predicted')
        t=np.arange(len(filtered[i][0,:]))
        plt.plot(t, filtered[i][0,:], 'g', label='filtered')
        plt.legend()
        plt.show()

        rmse_f=rmse(measured[0,:], filtered[i][0,:len(measured[0,:])])
        rmse_p=rmse(measured[0,:], predicted[i][0,:len(measured[0,:])])

        print("measured vs filtered RMSE for ",titles[i],": ",rmse_f)
        print("measured vs predicted RMSE for ",titles[i],": ",rmse_p)

        plt.xlabel('t')
        plt.ylabel('Position y')
        plt.suptitle(titles[i])
        t=np.arange(len(contaminated[i][1,:]))
        plt.plot(t, contaminated[i][1,:], 'r', label='contaminated')
        t=np.arange(measured.shape[1])
        plt.plot(t, measured[1,:], label='measured')
        plt.fill_between(t, sd_y, alpha=0.2)
        t=np.arange(len(predicted[i][1,:]))
        plt.plot(t, predicted[i][1,:], 'b', label='predicted')
        t=np.arange(len(filtered[i][1,:]))
        plt.plot(t, filtered[i][1,:], 'g', label='filtered')
        plt.legend()
        plt.show()
        
        rmse_f=rmse(measured[1,:], filtered[i][1,:len(measured[1,:])])
        rmse_p=rmse(measured[1,:], predicted[i][1,:len(measured[1,:])])

        print("measured vs filtered RMSE for ",titles[i],": ",rmse_f)
        print("measured vs predicted RMSE for ",titles[i],": ",rmse_p)

def differentN(measured, estimated, predicted):
    rmse_e=[]
    rmse_p=[]
    N=[]

    for i in range(measured.shape[1]+1,302, -15):
        N.append(i)
        e=rmse(measured[:i], estimated[:i])
        p=rmse(measured[:i], predicted[:i])
        rmse_e.append(e)
        rmse_p.append(p)

    plt.xlabel('N')
    plt.ylabel('RMSE')
    plt.plot(N, rmse_e, label='estimated')
    plt.plot(N, rmse_p, label='predicted')
    plt.legend()
    plt.show()


def missingObH():
    measured, contaminated, predicted, filtered, total, sd_y=main(fileName='missing_ball.mp4', missing_handled='yes')

    plotPositions(measured, filtered, predicted, contaminated, total, sd_y)
    
    measured2 = measured[np.logical_not(np.isnan(measured))]
    filtered = filtered[np.logical_not(np.isnan(measured))]
    predicted = predicted[np.logical_not(np.isnan(measured))]
    
    rmse_f=rmse(measured2[0:], filtered[0:])
    rmse_p=rmse(measured2[0:], predicted[0:])

    print("Missing object handled measured vs filtered RMSE: ", rmse_f)
    print("Missing object handled measured vs predicted RMSE: ", rmse_p)
    

def missingObNH():
    measured, contaminated, predicted, filtered, total, sd_y=main(fileName='missing_ball.mp4')

    plotPositions(measured, filtered, predicted, contaminated, total, sd_y)
    
    measured = measured[np.logical_not(np.isnan(measured))]
    predicted = predicted[np.logical_not(np.isnan(predicted))]
    filtered = filtered[np.logical_not(np.isnan(filtered))]
    
    rmse_f=rmse(measured[0:], filtered[0:])
    rmse_p=rmse(measured[0:], predicted[0:])

    print("Missing object not handled measured vs filtered RMSE: ", rmse_f)
    print("Missing object not handled measured vs predicted RMSE: ", rmse_p)



def plotPositions(measured, filtered, predicted, contaminated, total, sd_y):
    t=np.linspace(0, 1, total)
    
    plt.suptitle('Positions')
    plt.xlabel('time')
    plt.ylabel('position y')

    
    plt.plot(t, contaminated[1,:], label='contaminated', c='r', linewidth=0.5)
    plt.plot(t, measured[1,:], label='measured', c='b', linewidth=0.5)
    plt.fill_between(t, sd_y, alpha=0.2)
    plt.plot(t, predicted[1,:], label='predicted', c='k', linewidth=0.5)
    plt.plot(t, filtered[1,:], label='filtered', c='g', linewidth=0.5)
    plt.legend()
    plt.show()

    plt.suptitle('Positions')
    plt.xlabel('time')
    plt.ylabel('position x')

    plt.plot(t, contaminated[0,:], label='contaminated', c='r', linewidth=0.5)
    plt.plot(t, measured[0,:], label='measured', c='b', linewidth=0.5)
    plt.fill_between(t, sd_y, alpha=0.2)
    plt.plot(t, predicted[0,:], label='predicted', c='k', linewidth=0.5)
    plt.plot(t, filtered[0,:], label='filtered', c='g', linewidth=0.5)
    plt.legend()
    plt.show()
    

if __name__ == "__main__":

    differentNoises()
    
    differentAcc()

    missingObNH()

    missingObH()

    measured, contaminated, predicted, filtered, total, sd_y=main()

    differentN(measured, estimated, predicted)
    

    plotPositions(measured, filtered, predicted, contaminated, total, sd_y)
    

    

    
    
