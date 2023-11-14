let pointsToDraw = [];
let distanceBetweenPoints = [];

let highlightedLineIndexOnClick = -1;
let highlightedLineIndexOnHover = -1;

window.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");
    const imgSrc = "https://upload.wikimedia.org/wikipedia/commons/b/b4/Crossroads_in_front_of_Shimonoseki_City_Office.jpg";

    // Load the image on the canvas
    const img = new Image();
    img.src = imgSrc;

    img.onload = () => {
        context.drawImage(img, 0, 0, canvas.width, canvas.height);
    };

    // Function to handle mouse clicks and draw points
    canvas.addEventListener("click", (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        highlightedLineIndexOnClick = getHighlightedLineIndex(x, y);
        // State, when is clicked for new point (new point added)
        if (highlightedLineIndexOnClick == -1) {

            if (pointsToDraw.length < 4) {
                pointsToDraw.push({ x, y });
                if (pointsToDraw.length > 1) {
                    distanceBetweenPoints.push(0);
                }
            }
            else {
                pointsToDraw.pop();
                distanceBetweenPoints.pop();
                pointsToDraw.push({ x, y });
                distanceBetweenPoints.push(0);
            }
            // Connect last added point with first point
            if (pointsToDraw.length >= 3) {
                pointsToDraw.push(pointsToDraw[0]);
                distanceBetweenPoints.push(0);
            }
        }
        // State, when line is clicked
        else {
            document.getElementById("pointsDistanceValue").value = distanceBetweenPoints[highlightedLineIndexOnClick];
            //console.log(distanceBetweenPoints[highlightedLineIndexOnClick].toString())
            showModal();
        }
        drawPointsOnCanvas();
    });

    // Function to handle mousemove and highlight lines
    canvas.addEventListener("mousemove", (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        highlightedLineIndexOnHover = getHighlightedLineIndex(x, y);
        drawPointsOnCanvas();
    });

    function drawPointsOnCanvas() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Draw points on the canvas
        context.fillStyle = "red";
        context.font = "normal normal normal 14px arial";
        pointIndex = 0;

        for (let i = 0; i < pointsToDraw.length; i++) {
            context.beginPath();
            context.arc(pointsToDraw[i].x, pointsToDraw[i].y, 5, 0, 2 * Math.PI);
            context.fill();

            // Connect last added point with first point
            if (pointsToDraw.length >= 3) {
                if (i == (pointsToDraw.length - 1)) {
                    break;
                }
            }
            // Draw point labels
            context.fillText("P " + pointIndex.toString(), pointsToDraw[i].x, pointsToDraw[i].y - 10);
            pointIndex = pointIndex + 1;
        }

        // Draw lines between consecutive points
        if (pointsToDraw.length > 1) {
            context.strokeStyle = "blue";
            context.lineWidth = 2;
            context.beginPath();
            context.moveTo(pointsToDraw[0].x, pointsToDraw[0].y);
            for (let i = 1; i < pointsToDraw.length; i++) {
                context.lineTo(pointsToDraw[i].x, pointsToDraw[i].y);
            }
            context.stroke();
        }

        // Highlight the line on hover
        if (highlightedLineIndexOnHover !== -1 && pointsToDraw.length > 1) {
            const { x: mouseX, y: mouseY } = pointsToDraw[highlightedLineIndexOnHover] || {};
            const { x: nextX, y: nextY } = pointsToDraw[highlightedLineIndexOnHover + 1] || {};

            context.strokeStyle = "green";
            context.lineWidth = 4;
            context.beginPath();
            context.moveTo(mouseX, mouseY);
            context.lineTo(nextX, nextY);
            context.stroke();
        }

        // Highlight the line on click
        if (highlightedLineIndexOnClick !== -1 && pointsToDraw.length > 1) {
            const { x: mouseX, y: mouseY } = pointsToDraw[highlightedLineIndexOnClick] || {};
            const { x: nextX, y: nextY } = pointsToDraw[highlightedLineIndexOnClick + 1] || {};

            context.strokeStyle = "green";
            context.lineWidth = 8;
            context.beginPath();
            context.moveTo(mouseX, mouseY);
            context.lineTo(nextX, nextY);
            context.stroke();
        }
    }

    function getHighlightedLineIndex(x, y) {
        let closestDistance = 15;
        let index = -1;

        for (let i = 0; i < pointsToDraw.length - 1; i++) {
            const { x: x1, y: y1 } = pointsToDraw[i];
            const { x: x2, y: y2 } = pointsToDraw[i + 1];
            const distance = pointToLineDistance(x, y, x1, y1, x2, y2);

            if (distance < closestDistance) {
                closestDistance = distance;
                index = i;
            }
        }
        return index;
    }

    function pointToLineDistance(x, y, x1, y1, x2, y2) {
        const A = x - x1;
        const B = y - y1;
        const C = x2 - x1;
        const D = y2 - y1;

        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        let param = -1;

        if (lenSq !== 0) {
            param = dot / lenSq;
        }

        let xx, yy;

        if (param < 0) {
            xx = x1;
            yy = y1;
        } else if (param > 1) {
            xx = x2;
            yy = y2;
        } else {
            xx = x1 + param * C;
            yy = y1 + param * D;
        }

        const dx = x - xx;
        const dy = y - yy;

        return Math.sqrt(dx * dx + dy * dy);
    }

    const clearLastPointBtnElement = document.getElementById("clearLastPointBtn");
    clearLastPointBtnElement.addEventListener("click", function () {
        drawPointsOnCanvas();
    });

    const clearAllPointsBtnElement = document.getElementById("clearAllPointsBtn");
    clearAllPointsBtnElement.addEventListener("click", function () {
        drawPointsOnCanvas();
    });
});

function clearLastPoint() {
    pointsToDraw.pop();
    distanceBetweenPoints.pop();
    if (pointsToDraw.length >= 3) {
        pointsToDraw.pop();
        distanceBetweenPoints.pop();

        if (pointsToDraw.length > 2) {
            pointsToDraw.push(pointsToDraw[0]);
            distanceBetweenPoints.push(0);
        }
    }
    highlightedLineIndexOnClick = highlightedLineIndexOnClick - 1;
    highlightedLineIndexOnHover = highlightedLineIndexOnHover - 1;
}

function clearAllPoints() {
    pointsToDraw.length = 0;
    distanceBetweenPoints.length = 0;
    highlightedLineIndexOnClick = highlightedLineIndexOnClick -1;
    highlightedLineIndexOnHover = highlightedLineIndexOnHover - 1;
}

// Function to show the modal popup
function showModal() {
    const modalPopup = document.getElementById("modalPopup");
    modalPopup.style.display = "block";
}

// Function to close the modal popup
function closeModal() {
    const modalPopup = document.getElementById("modalPopup");
    modalPopup.style.display = "none";
    distanceBetweenPoints[highlightedLineIndexOnClick] = document.getElementById("pointsDistanceValue").value;
    //console.log(document.getElementById("pointsDistanceValue").value);
    //console.log(highlightedLineIndexOnClick)
}
