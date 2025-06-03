import cv2
import numpy as np
import supervision as sv
from trackers import SORTTracker
from tensorrtforge import pybind_tracker


tracker = SORTTracker()
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
trace_annotator = sv.TraceAnnotator()


def deep_sort_compute(image, results):
    #print("Deep SORT compute function called")
    #print("Image shape:", image.shape)  
    
    
    
    count = results.size
    
    if count:
        bboxes = np.array(results.bbox_array).reshape(-1, 4).astype(np.int32)
        conf = np.array(results.conf_array)
        class_id = np.array(results.class_id_array).astype(int)
        
        
        det = sv.Detections(
            xyxy=bboxes,
            confidence=conf,
            class_id=class_id,
        )

        det = tracker.update(det)

        annotator.annotate(image, det, labels=det.tracker_id)
        #trace_annotator.annotate(image, det)
        
    #print("Detection count:", count)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting the program.")
        return False
    
    cv2.imshow("Image", image)
    cv2.waitKey(1)
    
    return True
    
    
    