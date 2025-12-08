class ECGZoomHandler {
    constructor(plotElement) {
        this.plot = plotElement;
        this.isZooming = false;
        this.zoomStart = null;
        this.currentTimeRange = [0, 10]; // Default 10-second view
        
        this.initializeZoom();
    }

    initializeZoom() {
        this.plot.addEventListener('mousedown', (e) => this.startZoom(e));
        this.plot.addEventListener('mousemove', (e) => this.updateZoom(e));
        this.plot.addEventListener('mouseup', (e) => this.endZoom(e));
        this.plot.addEventListener('mouseleave', () => this.cancelZoom());
    }

    startZoom(event) {
        this.isZooming = true;
        this.zoomStart = this.getTimeFromPosition(event.offsetX);
    }

    updateZoom(event) {
        if (!this.isZooming) return;
        
        const currentTime = this.getTimeFromPosition(event.offsetX);
        // Update visual feedback for zoom selection
        // Implementation depends on your plotting library
    }

    endZoom(event) {
        if (!this.isZooming) return;
        
        const zoomEnd = this.getTimeFromPosition(event.offsetX);
        this.isZooming = false;

        // Ensure proper order of time range
        const startTime = Math.min(this.zoomStart, zoomEnd);
        const endTime = Math.max(this.zoomStart, zoomEnd);

        if (endTime - startTime < 0.1) { // Minimum zoom window of 100ms
            return;
        }

        this.currentTimeRange = [startTime, endTime];
        this.refreshDisplay();
    }

    cancelZoom() {
        this.isZooming = false;
        this.zoomStart = null;
    }

    getTimeFromPosition(x) {
        const plotWidth = this.plot.clientWidth;
        const timeRange = this.currentTimeRange[1] - this.currentTimeRange[0];
        return this.currentTimeRange[0] + (x / plotWidth) * timeRange;
    }

    resetZoom() {
        this.currentTimeRange = [0, 10];
        this.refreshDisplay();
    }

    async refreshDisplay() {
        // Fetch new data with zoom range and update display
        const segmentId = getCurrentSegmentId(); // Implement this based on your app
        const response = await fetch(`/api/segment/${segmentId}?start=${this.currentTimeRange[0]}&end=${this.currentTimeRange[1]}`);
        const data = await response.json();
        updatePlot(data); // Implement this based on your plotting library
    }
}
