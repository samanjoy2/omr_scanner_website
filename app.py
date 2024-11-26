import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display images
def display_image(image, title):
    st.image(image, caption=title, use_column_width=True)

def segment_document(image):
    """
    Segment the document from the background using GrabCut and extract the vertices.
    
    Args:
    image (np.array): Input image

    Returns:
    np.array: Segmented document
    np.array: Vertices of the segmented document
    """
    # Create a mask and initialize models
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run GrabCut
    rect = (20, 20, image.shape[1] - 40, image.shape[0] - 40)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Create a binary mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = image * mask2[:, :, np.newaxis]

    # Find contours and extract vertices
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) > 0:
        # Get the largest contour and approximate its vertices
        largest_contour = contours[0]
        perimeter = cv2.arcLength(largest_contour, True)
        vertices = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)
        
        # Ensure the vertices are in the correct format (4 x 2)
        if len(vertices) == 4:
            vertices = vertices.reshape((4, 2))
        else:
            # If the contour has more/less than 4 vertices, use the image borders
            vertices = np.array([[10, 10], [image.shape[1]-10, 10], [image.shape[1]-10, image.shape[0]-10], [10, image.shape[0]-10]], dtype=np.float32)
    else:
        # If no contours are found, use the image borders
        vertices = np.array([[10, 10], [image.shape[1]-10, 10], [image.shape[1]-10, image.shape[0]-10], [10, image.shape[0]-10]], dtype=np.float32)

    return segmented, vertices

def reorder_vertices(vertices):
    """
    Reorder the vertices in the correct order (top-left, top-right, bottom-right, bottom-left).
    
    Args:
    vertices (np.array): Vertices of the segmented document

    Returns:
    np.array: Reordered vertices
    """
    # Sort the vertices based on their sum and difference
    reordered = np.zeros_like(vertices, dtype=np.float32)
    s = vertices.sum(axis=1)
    diff = np.diff(vertices, axis=1)

    reordered[0] = vertices[np.argmin(s)]  # Top-left
    reordered[2] = vertices[np.argmax(s)]  # Bottom-right
    reordered[1] = vertices[np.argmin(diff)]  # Top-right
    reordered[3] = vertices[np.argmax(diff)]  # Bottom-left

    return reordered

def crop_and_transform(image, vertices):
    """
    Crop and apply perspective transform to the segmented document.
    
    Args:
    image (np.array): Original input image
    vertices (np.array): Vertices of the segmented document

    Returns:
    np.array: Cropped and transformed document
    """
    # Reorder the vertices
    vertices = reorder_vertices(vertices)

    # Calculate the width and height of the transformed image
    (tl, tr, br, bl) = vertices
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct the destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Apply the perspective transform
    M = cv2.getPerspectiveTransform(vertices, dst)
    transformed = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return transformed

# Function to crop the image (remove upper, lower, left, and right portions)
def crop_image(image, top_fraction=0.3, bottom_fraction=0.01, left_fraction=0.0, right_fraction=0.0):
    height, width = image.shape[:2]
    
    # Calculate the pixel values to crop
    top = int(height * top_fraction)
    bottom = int(height * (1 - bottom_fraction))
    left = int(width * left_fraction)
    right = int(width * (1 - right_fraction))
    
    # Crop the image
    cropped_image = image[top:bottom, left:right]
    
    return cropped_image

# Function to split the image into four vertical parts
def split_image_into_four_vertical(image):
    height, width = image.shape[:2]
    # Calculate the width for each of the four parts
    part_width = width // 4
    
    # Split the image into four vertical parts
    parts = [
        image[:, 0:part_width],             # First part (left)
        image[:, part_width:2*part_width],  # Second part
        image[:, 2*part_width:3*part_width], # Third part
        image[:, 3*part_width:]             # Fourth part (right)
    ]
    
    return parts

def detect_bubbles_and_answers(image, min_fill_ratio=0.2, max_fill_ratio=0.8):
    """
    Detect bubbles and determine the marked answers based on fill ratios.
    
    Parameters:
    - image: Input image
    - min_fill_ratio: Minimum ratio of filled pixels to area to consider a bubble marked.
    - max_fill_ratio: Maximum ratio of filled pixels to area to consider a bubble valid.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding to enhance bubble edges
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 5
    )

    # Perform Hough Circle Transform (Detecting "bubbles")
    circles = cv2.HoughCircles(
        thresh,
        cv2.HOUGH_GRADIENT,
        dp=0.2,
        minDist=10,
        param1=10,
        param2=10,
        minRadius=4,
        maxRadius=9
    )

    result_img = image.copy()
    answers = {}
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Group circles by rows (based on y-coordinates)
        rows = {}
        for (x, y, r) in circles:
            row_found = False
            for row_y in rows:
                if abs(row_y - y) <= r * 2:  # Adjust grouping tolerance
                    rows[row_y].append((x, y, r))
                    row_found = True
                    break
            if not row_found:
                rows[y] = [(x, y, r)]

        # Process each row to find the marked answer
        for row_y in sorted(rows.keys()):
            row_bubbles = sorted(rows[row_y], key=lambda b: b[0])
            bubble_status = []

            for (x, y, r) in row_bubbles:
                # Create a mask for the bubble
                mask = np.zeros_like(gray)
                cv2.circle(mask, (x, y), r, 255, -1)
                roi = cv2.bitwise_and(thresh, thresh, mask=mask)

                # Calculate fill ratio
                total_area = np.pi * r**2
                filled_pixels = np.sum(roi == 255)
                fill_ratio = filled_pixels / total_area
                bubble_status.append((x, y, r, fill_ratio))

            # Determine the marked answer in the row (Fill ratio greater than 0.65)
            marked_bubble = None
            for idx, (x, y, r, fill_ratio) in enumerate(bubble_status):
                if fill_ratio > 0.57:  # If fill ratio is greater than 65%
                    marked_bubble = idx
                    break

            if marked_bubble is not None:
                answer = chr(65 + marked_bubble)  # Convert index to A, B, C, D
                if answer > "D":
                    answer = "D"
                answers[row_y] = answer
                cv2.circle(result_img, (row_bubbles[marked_bubble][0], row_bubbles[marked_bubble][1]), row_bubbles[marked_bubble][2], (255, 0, 0), 3)
                cv2.putText(result_img, answer, (row_bubbles[marked_bubble][0] - 10, row_bubbles[marked_bubble][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                answers[row_y] = "?"  # No answer marked for this row

    return result_img, answers, gray, blurred, thresh

def main():
    # Streamlit interface
    st.title("Document Processing and Answer Detection")
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        # Read the image using OpenCV
        image = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Display original image
        st.subheader("Original Image")
        display_image(image, "Original Image")
        
        # Process the image (segment document and transform)
        segmented, vertices = segment_document(image)
        transformed_document = crop_and_transform(image, vertices)

        st.subheader("Transformed Document")
        display_image(transformed_document, "Transformed Document")
        
        # Split the transformed image into four vertical parts
        parts = split_image_into_four_vertical(transformed_document)
        st.subheader("Cropped Parts")
        for i, part in enumerate(parts):
            display_image(part, f"Cropped Part {i + 1}")
        
        # Detect bubbles and answers from each cropped part
        all_answers = {}
        for i, part in enumerate(parts):
            result_img, answers, gray, blurred, thresh = detect_bubbles_and_answers(part)
            all_answers[i + 1] = answers
            st.subheader(f"Detected Answers for Part {i + 1}")
            display_image(result_img, f"Processed Part {i + 1}")
        
        # Display all detected answers
        st.subheader("Detected Answers")
        st.write(all_answers)

# Run the app
if __name__ == "__main__":
    main()
