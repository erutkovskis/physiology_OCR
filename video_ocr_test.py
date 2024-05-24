import cv2
import pytesseract
import re
import csv
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Adjust to your system Tesseract path

def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve OCR accuracy
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to binarize the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply dilation and erosion to highlight text features
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_image = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return processed_image

def format_number(text):
    """
    Function to format numbers by inserting a decimal point if missing.
    Assumes the decimal point is missing between the last two digits.
    """
    if re.fullmatch(r'\d{2}', text):
        return text[0] + '.' + text[1]
    return text

def extract_text_from_video(video_path, isVerbose, isPulm):
    """
    Function to extract numbers from video. 
    video_path - path
    isVerbose - flag to output recognised stuff to the terminal
    isPulm - flag to format respiratory rate correctly (comma-separated decimal)
    """
    cap = cv2.VideoCapture(video_path)
    
    #rows, cols = (2, 0)
    
    detected_data = {}
    #nrs_recognised = [[0 for i in range(cols)] for j in range(rows)]
    newValue = 0
    frameSkip = 15 # fps. Number of frames skipped for recognition.
    frame_count = 0 # frame count variable to skip some frames to reduce sampling rate

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
            
        # Preprocess the frame
        processed_frame = preprocess_image(frame)

        data = pytesseract.image_to_data(processed_frame, config=r'--oem 3 --psm 6',output_type=pytesseract.Output.DICT)
        n_boxes = len(data['level'])
        unique_id = 1  # Initialize unique identifier for numbers
        for i in range(n_boxes):
            text = data['text'][i]
            if re.fullmatch(r'\d+(\.\d+)?', text): # Check if the detected text is a digit or a decimal number
               if isPulm:
                   text = format_number(text)
               (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
               label = f'{unique_id}: {text}'
               frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
               frame = cv2.putText(frame, label, (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2) 
               
               newValue = text
            #    nrs_recognised[unique_id].append(newValue)
               if unique_id not in detected_data:
                    detected_data[unique_id] = []
               timestamp = (int(cap.get(cv2.CAP_PROP_POS_MSEC))/1000.0)
               detected_data[unique_id].append({'Time (s)': timestamp,'Recognised': newValue})
               if isVerbose:
                print("Timestamp (s): %f, 'Id:' %d,\nRecognised: %s" % (timestamp, unique_id, newValue)) 
               unique_id += 1 # increment unique number identifier 

        cv2.imshow('Number Recognition', frame)
        frame_count += frameSkip # set to reduce sampling freq
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        #print("Timestamp (ms): %d, \nRecognised: %s" % (timestamp, data))
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cap.release()
    cv2.destroyAllWindows() 
    
    return detected_data

def write_to_csv(path_to_csv,detected_data):

    # Write organized data to CSV file
    with open(path_to_csv, 'w', newline='') as csvfile:
        fieldnames = ['Time (s)'] + [f'Unique_Id_{uid}' for uid in detected_data.keys()]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        max_entries = max(len(entries) for entries in detected_data.values())
        
        for i in range(max_entries):
            row = {'Time (s)': None}
            for uid in detected_data.keys():
                if i < len(detected_data[uid]):
                    row[f'Unique_Id_{uid}'] = detected_data[uid][i]['Recognised']
                    row['Time (s)'] = detected_data[uid][i]['Time (s)']  # Use the timestamp from the first entry
                else:
                    row[f'Unique_Id_{uid}'] = None
            writer.writerow(row)
            


detected_data = extract_text_from_video('path/to/video.mp4',1,0)
#print_recognised(times,numbers)
write_to_csv('path/to/csv.csv',detected_data)
