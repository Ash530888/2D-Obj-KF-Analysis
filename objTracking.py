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
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import math

def main(fileName='video_randomball.avi', missing_handled='no', u_x=1, u_y=1, addNoise=False, Gamma=None, Sigma=None):

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

    if addNoise:
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

        if addNoise:
            for i in centers_true:
                centers.append(np.array([i[0]+w[0,matrixIndex],i[1]+v[0,matrixIndex]]))
        
        # If centroids are detected then track them
        if (len(centers_true) > 0):

            # Draw the detected circle
            cv2.circle(frame, (int(centers_true[0][0]), int(centers_true[0][1])), 10, (0, 191, 255), 2)

            # Predict
            (x, y) = KF.predict()
            # Draw a rectangle as the predicted object position

            cv2.rectangle(frame, (int(x - 30), int(y - 30)),(int( x + 30), int(y + 30)), (255, 0, 0), 2)
            # Update
            if addNoise:
                (x1, y1) = KF.update(centers[0])
            else:
                (x1, y1) = KF.update(centers_true[0])

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

            if addNoise:
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


def differentNoises():
    noises_x=[np.array([[100,0],[0,80]]), np.array([[5,0],[0,7]]), np.array([[1,0],[0,0.5]]),  np.array([[10,0],[0,9]]),  np.array([[50,0],[0,67]]),  np.array([[70,0],[0,20]])]
    noises_y=[np.array([[5,0],[0,5]]),np.array([[0.05,0],[0,0.05]]), np.array([[0.07,0],[0,0.07]]), np.array([[0.1,0],[0,0.1]]), np.array([[0.5,0],[0,0.5]]), np.array([[1,0],[0,1]])]

    rmse_x=[]
    rmse_y=[]
    
    for i in range(6):
        measured, contaminated, predicted, filtered, total, sd_y=main(addNoise=True, Gamma=noises_x[i], Sigma=noises_y[i])
        
        predicted = predicted[np.logical_not(np.isnan(predicted))].reshape(2,-1)
        filtered = filtered[np.logical_not(np.isnan(filtered))].reshape(2,-1)
        contaminated = contaminated[np.logical_not(np.isnan(contaminated))].reshape(2,-1)
        measured = measured[np.logical_not(np.isnan(measured))].reshape(2,-1)

        t=np.arange(total-1)
        plt.xlabel('time')
        plt.ylabel('Position x')
        plt.plot(t, contaminated[0,:], color='r', label='contaminated', linewidth=0.4)
        plt.plot(t, measured[0,:] , label='measured', linewidth=0.4)
        #plt.errorbar(t, measured[0,:], yerr=sd_y, label='measured', linewidth=0.4, ecolor='black')
        plt.plot(t, predicted[0,:], color='black', label='predicted', linewidth=0.4)
        plt.plot(t, filtered[0,:] , color='g', label='filtered', linewidth=0.4)
        plt.legend()
        plt.show()

        rmse_f=mean_squared_error(measured[0,:], filtered[0,:len(measured[0,:])], squared=False)
        rmse_p=mean_squared_error(measured[0,:], predicted[0,:len(measured[0,:])], squared=False)

        rmse_x.append(rmse_f)

        rmse_f=mean_squared_error(measured[1,:], filtered[1,:len(measured[0,:])], squared=False)

        rmse_y.append(rmse_f)

        print("measured vs filtered RMSE when noise_x= ",noises_x[i][0][0],": ",rmse_f)
        print("measured vs predicted RMSE when noise_x= ",noises_x[i][0][0],": ",rmse_p)

        noises_x[i]=noises_x[i][0][0]
        noises_y[i]=noises_y[i][0][0]

    plt.suptitle("RMSE vs noise")
    plt.xlabel("noise")
    plt.ylabel("RMSE")
    plt.scatter(noises_x, rmse_x, label="x")
    plt.scatter(noises_y, rmse_y, label="y")
    plt.legend()
    plt.show()



def differentN(measured, filtered, predicted):
    rmse_f=[]
    rmse_p=[]
    N=[]

    for i in range(measured.shape[1]+1,302, -15):
        N.append(i)
        f=mean_squared_error(measured[:i], filtered[:i], squared=False)
        p=mean_squared_error(measured[:i], predicted[:i], squared=False)
        rmse_f.append(f)
        rmse_p.append(p)

    plt.xlabel('N')
    plt.ylabel('RMSE')
    plt.plot(N, rmse_e, label='filtered')
    plt.plot(N, rmse_p, label='predicted')
    plt.legend()
    plt.show()

def differentAcc():
    measured, contaminated, predicted, filtered, total, sd_y=main()
    measured, contaminated1, predicted1, filtered1, total, sd_y=main(u_x=200,u_y=200)
    measured, contaminated2, predicted2, filtered2, total, sd_y=main(u_x=200,u_y=100)

    predicted=[predicted, predicted1, predicted2]
    filtered=[filtered, filtered1, filtered2]
    titles=["a=(1,1)", "a=(200,200)", "a=(200,100)"]

    measured = measured[np.logical_not(np.isnan(measured))].reshape(2,-1)
    predicted1 = predicted1[np.logical_not(np.isnan(predicted1))].reshape(2,-1)
    predicted2 = predicted2[np.logical_not(np.isnan(predicted2))].reshape(2,-1)
    filtered1 = filtered1[np.logical_not(np.isnan(filtered1))].reshape(2,-1)
    filtered2 = filtered2[np.logical_not(np.isnan(filtered2))].reshape(2,-1)

    for i in range(3):
        plt.xlabel('t')
        plt.ylabel('Position x')
        plt.suptitle(titles[i])
        t=np.arange(measured.shape[1])
        plt.plot(t, measured[0,:], label='measured')
        t=np.arange(len(predicted[i][0,:]))
        plt.plot(t, predicted[i][0,:], 'r', label='predicted')
        t=np.arange(len(filtered[i][0,:]))
        plt.plot(t, filtered[i][0,:], 'g', label='filtered')
        plt.legend()
        plt.show()

        rmse_f=mean_squared_error(measured[0,:], filtered[i][0,:len(measured[0,:])], squared=False)
        rmse_p=mean_squared_error(measured[0,:], predicted[i][0,:len(measured[0,:])], squared=False)

        print("measured vs filtered RMSE for ",titles[i],": ",rmse_f)
        print("measured vs predicted RMSE for ",titles[i],": ",rmse_p)

        plt.xlabel('t')
        plt.ylabel('Position y')
        plt.suptitle(titles[i])
        t=np.arange(measured.shape[1])
        plt.plot(t, measured[1,:], label='measured')
        t=np.arange(len(predicted[i][1,:]))
        plt.plot(t, predicted[i][1,:], 'r', label='predicted')
        t=np.arange(len(filtered[i][1,:]))
        plt.plot(t, filtered[i][1,:], 'g', label='filtered')
        plt.legend()
        plt.show()
        
        rmse_f=mean_squared_error(measured[1,:], filtered[i][1,:len(measured[1,:])], squared=False)
        rmse_p=mean_squared_error(measured[1,:], predicted[i][1,:len(measured[1,:])], squared=False)

        print("measured vs filtered RMSE for ",titles[i],": ",rmse_f)
        print("measured vs predicted RMSE for ",titles[i],": ",rmse_p)


def missingObH():
    measured, contaminated, predicted, filtered, total, sd_y=main(fileName='missing_ball.mp4', missing_handled='yes')

    plotPositions(measured, filtered, predicted,total)
    
    measured2 = measured[np.logical_not(np.isnan(measured))]
    filtered = filtered[np.logical_not(np.isnan(measured))]
    predicted = predicted[np.logical_not(np.isnan(measured))]
    
    rmse_f=mean_squared_error(measured2[0:], filtered[0:], squared=False)
    rmse_p=mean_squared_error(measured2[0:], predicted[0:], squared=False)

    print("Missing object handled measured vs filtered RMSE: ", rmse_f)
    print("Missing object handled measured vs predicted RMSE: ", rmse_p)
    

def missingObNH():
    measured, contaminated, predicted, filtered, total, sd_y=main(fileName='missing_ball.mp4')

    plotPositions(measured, filtered, predicted, total)
    
    measured = measured[np.logical_not(np.isnan(measured))]
    predicted = predicted[np.logical_not(np.isnan(predicted))]
    filtered = filtered[np.logical_not(np.isnan(filtered))]
    
    rmse_f=mean_squared_error(measured[0:], filtered[0:], squared=False)
    rmse_p=mean_squared_error(measured[0:], predicted[0:], squared=False)

    print("Missing object not handled measured vs filtered RMSE: ", rmse_f)
    print("Missing object not handled measured vs predicted RMSE: ", rmse_p)



def plotPositions(measured, filtered, predicted, total):
    t=np.linspace(0, 1, total)
    
    plt.suptitle('Positions')
    plt.xlabel('time')
    plt.ylabel('position y')

    plt.plot(t, measured[1,:], label='measured', c='b', linewidth=1.5)
    plt.plot(t, predicted[1,:], label='predicted', c='r', linewidth=0.5)
    plt.plot(t, filtered[1,:], label='filtered', c='g', linewidth=0.5)
    plt.legend()
    plt.show()

    plt.suptitle('Positions')
    plt.xlabel('time')
    plt.ylabel('position x')

    plt.plot(t, measured[0,:], label='measured', c='b', linewidth=1.5)
    plt.plot(t, predicted[0,:], label='predicted', c='r', linewidth=0.5)
    plt.plot(t, filtered[0,:], label='filtered', c='g', linewidth=0.5)
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    
    differentNoises()

    measured, contaminated, predicted, filtered, total, sd_y=main()

    plotPositions(measured, filtered, predicted, total)

    missingObNH()
    missingObH()
    
    differentAcc()
    
    
    differentN(measured, filtered, predicted)
    

    
    
