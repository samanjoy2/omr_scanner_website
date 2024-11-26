import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_image(image, top_fraction=0.3, bottom_fraction=0.01, left_fraction=0.0, right_fraction=0.0):
    """
    Crop the image by specified fractions from top, bottom, left, and right.
    """
    height, width = image.shape[:2]
    
    top = int(height * top_fraction)
    bottom = int(height * (1 - bottom_fraction))
    left = int(width * left_fraction)
    right = int(width * (1 - right_fraction))
    
    cropped_image = image[top:bottom, left:right]
    
    return cropped_image

def split_image_into_four_vertical(image):
    """
    Split the image into four equal vertical parts.
    """
    height, width = image.shape[:2]
    part_width = width // 4
    
    parts = [
        image[:, 0:part_width],
        image[:, part_width:2*part_width],
        image[:, 2*part_width:3*part_width],
        image[:, 3*part_width:]
    ]
    
    return parts

def detect_bubbles_and_answers(image, min_fill_ratio=0.2, max_fill_ratio=0.8):
    """
    Detect bubbles and determine the marked answers based on fill ratios.
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

    return result_img, answers

def merge_cropped_images_into_main(original_image, cropped_images):
    """
    Merge processed cropped images back into the main image.
    """
    height, width = original_image.shape[:2]
    merged_image = original_image.copy()
    part_width = width // 4

    for i, cropped in enumerate(cropped_images):
        cropped_resized = cv2.resize(cropped, (part_width, height))
        x_offset = i * part_width
        merged_image[:, x_offset:x_offset + part_width] = cropped_resized

    return merged_image

def main():
    st.title("Document Bubble Detection App")

    # File uploader
    uploaded_file = st.file_uploader("Choose a document image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Crop the image
        cropped_image = crop_image(original_image, top_fraction=0.32, bottom_fraction=0.026, 
                                   left_fraction=0.06, right_fraction=0.06)

        # Split into four parts
        parts = split_image_into_four_vertical(cropped_image)

        # Process each part for bubble detection
        processed_parts = []
        all_answers = {}
        for i, part in enumerate(parts):
            result_img, answers = detect_bubbles_and_answers(part)
            processed_parts.append(result_img)
            
            # Collect answers for each part
            for row_y, answer in answers.items():
                all_answers[f"Part {i+1}, Row {len(all_answers) + 1}"] = answer

        # Merge the cropped and processed parts back into the main image
        final_image = merge_cropped_images_into_main(original_image, processed_parts)

        # Display results
        st.subheader("Original Image")
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.subheader("Processed Image with Bubble Detection")
        st.image(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Display answers
        st.subheader("Detected Answers")
        for key, value in all_answers.items():
            st.write(f"{key}: {value}")

if __name__ == "__main__":
    main()