import cv2
import pytesseract
import re
import csv
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_video(video_path):

    cap = cv2.VideoCapture(video_path)
    timestamps = []
    nrs_recognised = []
    newValue = 0
    prevValue = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # limit the frame manually?
        # Rotate the image to improve Tesseract recognition?
        
        #text = pytesseract.image_to_string(gray, config='digits')
        data = pytesseract.image_to_data(gray, config=r'--oem 3 --psm 6 outputbase digits',output_type=pytesseract.Output.DICT)
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            if re.search(r'\d', data['text'][i]): # Check if the detected text is a digit
               (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
               frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
               frame = cv2.putText(frame, data['text'][i], (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2) 
               newValue = int(data['text'][i])
               if newValue != prevValue:
                   nrs_recognised.append(newValue)
                   prevValue = newValue
                   timestamps.append(int(cap.get(cv2.CAP_PROP_POS_MSEC)))
               else:
                   nrs_recognised.append(prevValue)
                   timestamps.append(int(cap.get(cv2.CAP_PROP_POS_MSEC)))
               print("Timestamp (ms): %d, \nRecognised: %s" % (timestamps[-1], nrs_recognised[-1])) 

        cv2.imshow('Number Recognition', frame)
        
        #print("Timestamp (ms): %d, \nRecognised: %s" % (timestamp, data))
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cap.release()
    cv2.destroyAllWindows() 
    
    return timestamps, nrs_recognised

def print_recognised(time_list,number_list):
    for count,value in enumerate(time_list):
        print("Timestamp (ms): %d, \nRecognised: %s" % (value, number_list[count]))  
    

def write_to_csv(path_to_csv,time_list,number_list):
    with open(path_to_csv,'w',newline='') as file:
        writer = csv.writer(file)
        field = ["Timestamp (ms)","Number recognised"]
        writer.writerow(field)
        for time_list_count,time_list_value in enumerate(time_list):
            writer.writerow([time_list_value,number_list[time_list_count]])



times,numbers = extract_text_from_video('./phys_test_4.mp4')
#print_recognised(times,numbers)
write_to_csv('./test_output_csv_1.csv',times,numbers)