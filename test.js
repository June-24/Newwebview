body {
    font-family: Arial, sans-serif;
}

.tooltip-button {
    position: relative;
    display: inline-block;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    border: 1px solid #ccc;
    background-color: #f8f8f8;
}

.tooltip-text {
    visibility: hidden;
    width: 160px;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px 0;
    position: absolute;
    z-index: 1;
    bottom: 125%; /* Position the tooltip above the button */
    left: 50%;
    margin-left: -80px; /* Use half of the tooltip width to center it */
    opacity: 0;
    transition: opacity 0.3s;
}

.tooltip-button:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}
