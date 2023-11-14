let pointsToDraw = [];
let distanceBetweenPoints = [];

let highlightedLineIndexOnClick = -1;
let highlightedLineIndexOnHover = -1;
let defaultBirdViewFilePath = "";
let img = new Image();

// Video stream
let image = new Image();
const targetWidth = 800; // Target width (adjust as needed)
const targetHeight = 450; // Target height (adjust as needed)

let selectedCameraViews = [];


window.addEventListener("DOMContentLoaded", () => {

    for (let i = 0; i < 4; i++) {
        selectedCameraViews.push(new Image());
    }

    console.log("Javascript loaded!");

    // Button for run processing
    document.getElementById("runButton").addEventListener("click", function () {
        console.log("button click");
        // Send an AJAX request to the Django view
        fetch("{% url 'run_external_app' %}", {
            method: 'POST',
        })
            .then(response => response.text())
            .then(data => {
                alert(data);
            })
            .catch(error => {
                alert("Error: " + error);
            });
    });

    // Websocket handling, incoming video stream
    const canvasVideoStream = document.getElementById('videoStreamCanvas');
    const contextVideoStream = canvasVideoStream.getContext('2d');
    //var udpSocket = null;
    try {
        udpSocket = new WebSocket('ws://127.0.0.1:8050');

        if (udpSocket.readyState === WebSocket.CONNECTING) {
            console.log('WebSocket connection is in the process of opening');
            const defaultCanvasImage = canvasVideoStream.getAttribute("src");
            image.src = defaultCanvasImage;

            image.onload = () => {
                contextVideoStream.drawImage(image, 0, 0, canvasVideoStream.width, canvasVideoStream.height);
            };
        }

        udpSocket.onmessage = (event) => {
            const frame_data_base64 = event.data; // Assuming the data received is base64-encoded
            const image = new Image();

            image.onload = () => {
                contextVideoStream.drawImage(image, 0, 0, canvasVideoStream.width, canvasVideoStream.height);
            };

            image.src = 'data:image/jpeg;base64,' + frame_data_base64;
        };
    } catch (error) {
        console.log(error);
    }

    // Horizontal chart handling
    var horizontalBarChartContext = document.getElementById("horizontalBarChart").getContext('2d');
    // Define data for the horizontal bar chart
    var data = {
        labels: ["Person", "Car", "Bus", "Truck", "Train/Tram"],
        datasets: [{
            data: [0, 0, 0, 0, 0], // Update with the actual data
            backgroundColor: [
                'rgba(15, 150, 255, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(255, 99, 132, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(153, 102, 255, 0.2)'
            ], // Bar colors
            borderColor: [
                'rgba(15, 15, 255, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(255, 99, 132, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(153, 102, 255, 1)'
            ], // Border colors
            borderWidth: 1 // Border width
        }]
    };
    // Create a bar chart and set the axis to be horizontal
    var myHorizontalBarChart = new Chart(horizontalBarChartContext, {
        type: 'bar',
        data: data,
        options: {
            indexAxis: 'y', // Set the axis to be horizontal (y-axis)
            scales: {
                x: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false, // Set to true to display the legend
                }
            }
        }
    });

    // Websocket handling for horizontal chart
    const socket = new WebSocket('ws://127.0.0.1:8051');
    socket.onmessage = (event) => {
        const newData = JSON.parse(event.data);
        myHorizontalBarChart.data.labels = newData.labels;
        myHorizontalBarChart.data.datasets[0].data = newData.data;
        myHorizontalBarChart.update();
    };

    // Camera view handling
    const canvas01 = document.getElementById("canvas01");
    const context01 = canvas01.getContext("2d");

    const canvas02 = document.getElementById("canvas02");
    const context02 = canvas02.getContext("2d");

    const canvas03 = document.getElementById("canvas03");
    const context03 = canvas03.getContext("2d");

    const canvas04 = document.getElementById("canvas04");
    const context04 = canvas04.getContext("2d");

    const canvasSelectedCameraView = document.getElementById("canvasSelectedCameraView");
    const contextSelectedCameraView = canvasSelectedCameraView.getContext("2d");

    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");
    //const imgSrc = "https://www.icesi.edu.co/mgis-portal-conocimiento/wp-content/uploads/2019/04/default.png";
    //const imgSrc = "{% static 'images/no_image.jpg' %}";

    // Load the image on the canvas
    const defaultCanvasImage = canvas.getAttribute("src");
    img.src = defaultCanvasImage;
    img.onload = () => {
        context.drawImage(img, 0, 0, canvas.width, canvas.height);

        context01.drawImage(img, 0, 0, canvas01.width, canvas01.height);
        context02.drawImage(img, 0, 0, canvas02.width, canvas02.height);
        context03.drawImage(img, 0, 0, canvas03.width, canvas03.height);
        context04.drawImage(img, 0, 0, canvas04.width, canvas04.height);

        contextSelectedCameraView.drawImage(img, 0, 0, canvasSelectedCameraView.width, canvasSelectedCameraView.height);
    };

    function convertCoordinatesToBiggerImage(
        x, // X-coordinate in the smaller image frame
        y, // Y-coordinate in the smaller image frame
        smallerWidth, // Width of the smaller image frame
        smallerHeight, // Height of the smaller image frame
        biggerWidth, // Width of the bigger image frame
        biggerHeight // Height of the bigger image frame
    ) {
        const scaleX = biggerWidth / smallerWidth;
        const scaleY = biggerHeight / smallerHeight;
        const biggerX = x * scaleX;
        const biggerY = y * scaleY;
        return { x: biggerX, y: biggerY };
    }

    // Function to handle mouse clicks and draw points
    canvas.addEventListener("click", (event) => {
        var fullSizeImageWidth = 1920;
        var fullSizeImageHeight = 1080;
        var resizedImageWidth = 800;
        var resizedImageHeight = 450;

        if (defaultBirdViewFilePath != "") {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            const scaleX = fullSizeImageWidth / resizedImageWidth;
            const scaleY = fullSizeImageHeight / resizedImageHeight;
            const fullSizeImagePointX = x * scaleX;
            const fullSizeImagePointY = y * scaleY;
            console.log(fullSizeImagePointX, fullSizeImagePointY);

            highlightedLineIndexOnClick = getHighlightedLineIndex(x, y);
            // State, when is clicked for new point (new point added)
            if (highlightedLineIndexOnClick == -1) {

                if (pointsToDraw.length < 4) {
                    pointsToDraw.push({ x, y });
                    if (pointsToDraw.length > 1) {
                        distanceBetweenPoints.push(0);
                        addNewPointToGroupList("0", distanceBetweenPoints.length, 0);
                    }
                }
                else {
                    pointsToDraw.pop();
                    distanceBetweenPoints.pop();
                    pointsToDraw.push({ x, y });
                    distanceBetweenPoints.push(0);
                    //addNewPointToGroupList("0", distanceBetweenPoints.length, 1);
                    editPointInGroupList("0", pointsToDraw.length - 2);           
                }

                // Draw a total value of selected points
                setTotalValueToGroupList(pointsToDraw.length);

                // Connect last added point with first point
                if (pointsToDraw.length >= 3) {
                    pointsToDraw.push(pointsToDraw[0]);
                    distanceBetweenPoints.push(0);
                    addNewPointToGroupList("0", distanceBetweenPoints.length, 1);
                    //addNewPointToGroupList("0", distanceBetweenPoints.length);
                }
            }
            // State, when line is clicked
            else {
                document.getElementById("pointsDistanceValue").value = distanceBetweenPoints[highlightedLineIndexOnClick];
                //console.log(distanceBetweenPoints[highlightedLineIndexOnClick].toString())
                showModal();
            }
            drawPointsOnCanvas();
        }
    });

    // Function to handle mousemove and highlight lines
    canvas.addEventListener("mousemove", (event) => {
        if (defaultBirdViewFilePath != "") {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            highlightedLineIndexOnHover = getHighlightedLineIndex(x, y);
            drawPointsOnCanvas();
        }
    });

    function drawPointsOnCanvas() {
        if (defaultBirdViewFilePath != "") {
            context.clearRect(0, 0, canvas.width, canvas.height);
            context.drawImage(img, 0, 0, canvas.width, canvas.height);

            // Draw points on the canvas
            context.fillStyle = "red";
            context.font = "normal normal normal 14px arial";
            pointIndex = 1;

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
        setTotalValueToGroupList(pointsToDraw.length);
    });

    const clearAllPointsBtnElement = document.getElementById("clearAllPointsBtn");
    clearAllPointsBtnElement.addEventListener("click", function () {
        drawPointsOnCanvas();
        setTotalValueToGroupList(0);
    });


    // Get the input element for the number value
    const numberInput = document.getElementById("pointsDistanceValue");
    // Get the "Save" button element
    const saveNumberButton = document.getElementById("saveNumberButton");
    // Variable to store the selected number
    let selectedNumber;

    // Event listener for the "Save" button click
    saveNumberButton.addEventListener("click", () => {
        // Get the value from the input field and convert it to a number
        selectedNumber = parseInt(numberInput.value);

        // Check if the selected number is valid (not NaN)
        if (!isNaN(selectedNumber)) {
            // Do something with the selected number
            editPointInGroupList(selectedNumber.toString(), highlightedLineIndexOnClick);
            closeModal();     
        }
    });

    

    const addItemBtn = document.getElementById("addItemBtn");
    const newItemInput = document.getElementById("newItem");
    const selectedPointsItemList = document.getElementById("selectedPointsItemList");

    // Add event listener for the input change event
    imageInputBirdView.addEventListener("change", function () {
        defaultBirdViewFilePath = imageInputBirdView.files[0];
        //console.log("Path:", imageInputBirdView.files[0]);
        if (defaultBirdViewFilePath) {
            const reader = new FileReader();
            reader.onload = function (e) {
                //img = new Image();
                img.onload = function () {
                    context.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(defaultBirdViewFilePath);
        }
    });

    // Add event listener for the input change event
    imageInput01.addEventListener("change", function () {
        let imagePath = imageInput01.files[0];
        console.log("Path:", imageInput01.files[0]);

        // Create an Image element
        const imgInput = new Image();

        if (imagePath) {
            console.log(imagePath);
            const reader = new FileReader();
            reader.onload = function (e) {
                //img = new Image();
                imgInput.onload = function () {
                    context01.drawImage(imgInput, 0, 0, canvas01.width, canvas01.height);
                };
                imgInput.src = e.target.result;
                selectedCameraViews[0] = imgInput;
            };
            reader.readAsDataURL(imagePath);
        }      
    });

    // Event listener for the "Save" button click
    canvas01.addEventListener("click", () => {
        //let defaultCanvasImage = canvas.getAttribute("src");
        //let imggg = new Image();
        //imggg.src = defaultCanvasImage;
        //const dataURL = canvas01.toDataURL("image/jpeg", 1.0);

        // Create an Image element
        //const image = new Image();

        // Set the image source to the data URL
        //image.src = dataURL;
        //image.src = selectedCameraViews[0]

        contextSelectedCameraView.drawImage(selectedCameraViews[0], 0, 0, canvasSelectedCameraView.width, canvasSelectedCameraView.height);
    });

    // Add event listener for the input change event
    imageInput02.addEventListener("change", function () {
        let imagePath = imageInput02.files[0];
        console.log("Path:", imageInput02.files[0]);

        // Create an Image element
        const imgInput = new Image();

        if (imagePath) {
            const reader = new FileReader();
            reader.onload = function (e) {
                //img = new Image();
                imgInput.onload = function () {
                    context02.drawImage(imgInput, 0, 0, canvas02.width, canvas02.height);
                };
                imgInput.src = e.target.result;
                selectedCameraViews[1] = imgInput;
            };
            reader.readAsDataURL(imagePath);
        }
    });

    // Event listener for the "Save" button click
    canvas02.addEventListener("click", () => {
        contextSelectedCameraView.drawImage(selectedCameraViews[1], 0, 0, canvasSelectedCameraView.width, canvasSelectedCameraView.height);
    });

    // Add event listener for the input change event
    imageInput03.addEventListener("change", function () {
        let imagePath = imageInput03.files[0];
        console.log("Path:", imageInput03.files[0]);

        // Create an Image element
        const imgInput = new Image();

        if (imagePath) {
            selectedCameraViews[2] = imagePath;
            const reader = new FileReader();
            reader.onload = function (e) {
                //img = new Image();
                imgInput.onload = function () {
                    context03.drawImage(imgInput, 0, 0, canvas03.width, canvas03.height);
                };
                imgInput.src = e.target.result;
                selectedCameraViews[2] = imgInput;
            };
            reader.readAsDataURL(imagePath);
        }
    });

    // Event listener for the "Save" button click
    canvas03.addEventListener("click", () => {
        contextSelectedCameraView.drawImage(selectedCameraViews[2], 0, 0, canvasSelectedCameraView.width, canvasSelectedCameraView.height);
    });

    // Add event listener for the input change event
    imageInput04.addEventListener("change", function () {
        let imagePath = imageInput04.files[0];
        console.log("Path:", imageInput04.files[0]);

        // Create an Image element
        const imgInput = new Image();

        if (imagePath) {
            selectedCameraViews[3] = imagePath;
            const reader = new FileReader();
            reader.onload = function (e) {
                //img = new Image();
                imgInput.onload = function () {
                    context04.drawImage(imgInput, 0, 0, canvas04.width, canvas04.height);
                };
                imgInput.src = e.target.result;
                selectedCameraViews[3] = imgInput;
            };
            reader.readAsDataURL(imagePath);
        }
    });

    // Event listener for the "Save" button click
    canvas04.addEventListener("click", () => {
        contextSelectedCameraView.drawImage(selectedCameraViews[3], 0, 0, canvasSelectedCameraView.width, canvasSelectedCameraView.height);
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

    //////////
    console.log(distanceBetweenPoints.length);
    const selectedPointsItemList = document.getElementById("selectedPointsItemList");

    if (distanceBetweenPoints.length > 2) {
        const lastItemForRemove = selectedPointsItemList.children[selectedPointsItemList.childElementCount - 1];
        selectedPointsItemList.removeChild(lastItemForRemove);

        let pointsDistance = distanceBetweenPoints[distanceBetweenPoints.length - 1];

        let pointsDistanceText = "Points P" + (selectedPointsItemList.childElementCount - 1).toString() + "-P1, " + "distance: " + pointsDistance.toString() + " cm";

        selectedPointsItemList.children[selectedPointsItemList.childElementCount - 1].innerText = pointsDistanceText;
        selectedPointsItemList.children[selectedPointsItemList.childElementCount - 1].value = pointsDistanceText;
    }
    else {
        let lastItemForRemove = selectedPointsItemList.children[selectedPointsItemList.childElementCount - 1];
        selectedPointsItemList.removeChild(lastItemForRemove);
        lastItemForRemove = selectedPointsItemList.children[selectedPointsItemList.childElementCount - 1];
        selectedPointsItemList.removeChild(lastItemForRemove);
    }

}

function clearAllPoints() {
    pointsToDraw.length = 0;
    distanceBetweenPoints.length = 0;
    highlightedLineIndexOnClick = highlightedLineIndexOnClick -1;
    highlightedLineIndexOnHover = highlightedLineIndexOnHover - 1;

    //const selectedPointsItemList = document.getElementById("selectedPointsItemList");
    //console.log(selectedPointsItemList.childElementCount);

    //for (let i = 0; i < selectedPointsItemList.childElementCount; i++) {
    //    console.log("som tu", i);
    //    let itemForRemove = selectedPointsItemList.children[selectedPointsItemList.childElementCount - 1];
    //    selectedPointsItemList.removeChild(itemForRemove);
    //}
    //console.log(selectedPointsItemList);
    selectedPointsItemList.innerHTML = "";
}

// Function to show the modal popup
function showModal() {
    //const modalPopup = document.getElementById("modalPopup");
    //modalPopup.style.display = "block";
    document.getElementById("pointsDistanceValue").value = distanceBetweenPoints[highlightedLineIndexOnClick];
    $('#numberModal').modal('show');
}

// Function to close the modal popup
function closeModal() {
    distanceBetweenPoints[highlightedLineIndexOnClick] = document.getElementById("pointsDistanceValue").value;
    $("#numberModal").modal("hide");
    //const modalPopup = document.getElementById("modalPopup");
    //modalPopup.style.display = "none";
    //distanceBetweenPoints[highlightedLineIndexOnClick] = document.getElementById("pointsDistanceValue").value;
    //console.log(document.getElementById("pointsDistanceValue").value);
    //console.log(highlightedLineIndexOnClick)
}

function setTotalValueToGroupList(newItemInput) {
    document.getElementById("totalSelectedPointsInput").value = newItemInput;
    //addNewPointToGroupList(newItemInput);
}

function addNewPointToGroupList(pointsDistance, index, connectPointWithFirstPoint) {
    const selectedPointsItemList = document.getElementById("selectedPointsItemList");
    let pointsDistanceText = "";
    if (connectPointWithFirstPoint == 0) {
        pointsDistanceText = "Points P" + index.toString() + "-P" + (index + 1).toString() + ", distance: " + pointsDistance.toString() + " cm";
    }
    else {
        pointsDistanceText = "Points P" + index.toString() + "-P1, " + "distance: " + pointsDistance.toString() + " cm";
    }
    const listItem = document.createElement("a");
    listItem.classList.add("list-group-item");
    listItem.classList.add("list-group-item-action");
    listItem.textContent = pointsDistanceText;

    selectedPointsItemList.appendChild(listItem);

    //console.log(selectedPointsItemList.children[1].innerText);
    //console.log(selectedPointsItemList);
    //selectedPointsItemList[0] = "aaaaa";
}

function editPointInGroupList(pointsDistance, index) {
    const selectedPointsItemList = document.getElementById("selectedPointsItemList");
    let pointsDistanceText = "Points P" + (index + 1).toString() + "-P" + (index + 2).toString() + ", distance: " + pointsDistance.toString() + " cm";
    selectedPointsItemList.children[index + 1].innerText = pointsDistanceText;
    selectedPointsItemList.children[index + 1].value = pointsDistanceText;
}
