import React, { useState, useEffect, useRef, useCallback } from 'react';
import { BrowserRouter, Routes, Route, Link, useNavigate } from 'react-router-dom';
import './index.css';
import ObjectDetectionDisplay from './ObjectDetectionDisplay';

// Splash Screen Component
const SplashScreen = () => {
  const navigate = useNavigate();
  
  useEffect(() => {
    // Redirect to dashboard after animation (3 seconds)
    const timer = setTimeout(() => {
      navigate('/dashboard');
    }, 3000);
    
    return () => clearTimeout(timer);
  }, [navigate]);
  
  return (
    <div className="splash-screen">
      <div className="splash-content">
        <div className="splash-logo"></div>
        <h1>SPOT<span>AGENT</span></h1>
        <div className="loading-bar">
          <div className="loading-progress"></div>
        </div>
        <p className="splash-subtitle">SYSTEM INITIALIZING</p>
        <Link to="/dashboard" className="skip-btn">SKIP</Link>
      </div>
    </div>
  );
};

// Dashboard Component
const Dashboard = () => {
  // Remove activeTab state
  // const [activeTab, setActiveTab] = useState('main');
  
  // Connection State (Keep New)
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false); // Keep this one for status
  const [reconnectAttempt, setReconnectAttempt] = useState(0);

  // Robot State (received from WebSocket - Keep New)
  const [status, setStatus] = useState("Disconnected");
  const [currentTaskPrompt, setCurrentTaskPrompt] = useState(null);
  const [lastThought, setLastThought] = useState(null);
  const [lastAction, setLastAction] = useState(null);
  const [lastActionParams, setLastActionParams] = useState(null);
  const [taskComplete, setTaskComplete] = useState(false);
  const [taskSuccess, setTaskSuccess] = useState(null);
  const [taskReason, setTaskReason] = useState(null);
  const [visionAnalysis, setVisionAnalysis] = useState(null); // Keep for potential background logging
  const [odometry, setOdometry] = useState(null);
  const [objectDetection, setObjectDetection] = useState(null);
  
  // Connect websocket (Keep New)
  const connectWebSocket = useCallback(() => {
    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws`;
    
    console.log(`Attempting to connect WebSocket (${reconnectAttempt + 1}): ${wsUrl}`);
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      setStatus("Connected - Idle");
      setReconnectAttempt(0);
      setSocket(ws); // Set the new socket state
    };
    
    ws.onmessage = (event) => {
      try {
        const newData = JSON.parse(event.data);
        // DEBUG: Log received data, specifically object_detection part - REMOVE
        /*
        if (newData.object_detection) {
            console.log('[App.jsx onmessage] Received object_detection:', {
                 status: newData.object_detection.status,
                 object_count: newData.object_detection.object_count,
                 base64_image_keys: newData.object_detection.base64_images ? Object.keys(newData.object_detection.base64_images) : 'None',
                 error: newData.object_detection.error
             });
        } else if ('object_detection' in newData) { // Log if key exists but is null/undefined
             console.log('[App.jsx onmessage] Received object_detection key but it was null/undefined');
        }
        */
        
        // Update state based on received data (using new state setters)
        if (newData.status !== undefined) setStatus(newData.status);
        if (newData.current_task_prompt !== undefined) setCurrentTaskPrompt(newData.current_task_prompt);
        if (newData.last_thought !== undefined) setLastThought(newData.last_thought);
        if (newData.last_action !== undefined) setLastAction(newData.last_action);
        if (newData.last_action_params !== undefined) setLastActionParams(newData.last_action_params);
        if (newData.task_complete !== undefined) setTaskComplete(newData.task_complete);
        if (newData.task_success !== undefined) setTaskSuccess(newData.task_success);
        if (newData.task_reason !== undefined) setTaskReason(newData.task_reason);
        if (newData.vision_analysis !== undefined) setVisionAnalysis(newData.vision_analysis);
        if (newData.odometry !== undefined) setOdometry(newData.odometry);
        if (newData.object_detection !== undefined) {
             // DEBUG: Log before setting state - REMOVE
             // console.log('[App.jsx onmessage] Setting objectDetection state.');
             setObjectDetection(newData.object_detection);
        }
      } catch (err) {
        console.error('[App.jsx onmessage] Parse error:', err.message); // Keep basic error log
      }
    };
    
    ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      setConnected(false);
      setStatus("Disconnected");
      setSocket(null); // Clear the socket state

      // Simple exponential backoff for reconnect attempts
      const nextReconnectDelay = Math.min(1000 * Math.pow(2, reconnectAttempt), 30000); // Max 30 seconds
      console.log(`Attempting reconnect in ${nextReconnectDelay / 1000} seconds...`);
      setTimeout(() => {
        setReconnectAttempt(prev => prev + 1);
        // connectWebSocket(); // This will be called by useEffect dependency change
      }, nextReconnectDelay);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      // onclose will handle the reconnect attempt
    };
    
    // Set the socket ref immediately for cleanup purposes
    socketRef.current = ws;

  }, [reconnectAttempt]); // Dependency remains correct
  
  // Effect to establish WebSocket connection on mount (Keep New)
  useEffect(() => {
    if (!connected && !socket) {
        console.log("useEffect: Triggering connectWebSocket");
        connectWebSocket();
    }
    // Cleanup function: Will run when the component unmounts OR
    // when dependencies change causing the effect to re-run (which we want to avoid here).
    return () => {
        // Use the ref to ensure we close the correct socket instance
        if (socketRef.current) { 
             console.log('useEffect Cleanup: Closing WebSocket connection.');
             // Prevent handlers from firing during intentional close
             socketRef.current.onclose = null; 
             socketRef.current.onerror = null;
             socketRef.current.close();
             socketRef.current = null; // Clear the ref after closing
        } else {
             console.log('useEffect Cleanup: No socket found in ref to close.');
        }
    };
  // **CRITICAL FIX:** Only depend on connectWebSocket (which depends on reconnectAttempt).
  // DO NOT include 'connected' or 'socket' here.
  }, [connectWebSocket]);

  // Ref to hold the current WebSocket instance for cleanup (Keep New)
  const socketRef = useRef(null);
  // Keep the socket instance in a ref for reliable access in cleanup
  useEffect(() => {
    socketRef.current = socket;
  }, [socket]);

  // Helper function to format numbers
  const formatNumber = (num, digits = 2) => {
    if (num === null || num === undefined || typeof num !== 'number') {
        return '---'; // Or 'N/A', or 0.00, depending on preference
    }
    return num.toFixed(digits);
  };

  // Load test image (Can be removed if not used, keep for now)
  const openTestImage = () => window.open('/images/test_image.jpg', '_blank');
  
  // Primary status determination (Use new state)
  const statusColor = connected ? 
    (status?.includes('Error') ? 'var(--error)' : 'var(--success)') : 
    'var(--warning)';
  
  // const statusText = data.status || (connected ? 'Connected' : 'Disconnected'); // Use new status state
  
  // --- JSX uses NEW state variables: status, currentTaskPrompt, etc. ---
  return (
    <div className="cyber-container dashboard-mode"> {/* Add class for TV mode */} 
      <header className="dashboard-header fixed-header"> {/* Keep header fixed? */} 
        <div className="logo-section">
            {/* <img src="/path/to/your/logo.png" alt="Logo" className="logo" /> Replace with actual logo */}
            <h1 className="main-title">SPOT AGENT</h1>
        </div>
        <div className="status-section">
            <div className="status-indicator" style={{backgroundColor: statusColor}}></div>
            <span className="status-text">{status}</span>
        </div>
      </header>

      {/* Remove Tab Navigation */}
      {/* <nav className="tabs"> ... </nav> */}

      {/* Main Content Area - Single View */}
      <main className="dashboard-content single-view"> {/* Add class for single view */} 
         
         {/* Use a grid layout to arrange panels */}
         <div className="main-layout-grid"> 
            
            {/* Column 1: Task & Action */}
            <div className="grid-column column-1">
                 {/* Task status panel */}
                 <div className="panel task-panel">
                   <div className="panel-header"><h2>CURRENT TASK</h2></div>
                   <div className="panel-body">
                     {currentTaskPrompt ? (
                       <div className="task-content">
                         <div className="task-prompt">{currentTaskPrompt}</div>
                         {taskComplete && (
                           <div className="task-outcome">
                             <div className={`outcome-result ${taskSuccess ? 'success' : 'failure'}`}>
                               {taskSuccess ? 'SUCCESS' : 'FAILURE'} {taskReason ? `- ${taskReason}` : ''}
                             </div>
                           </div>
                         )}
                       </div>
                     ) : (
                       <div className="no-task status-text waiting">AWAITING INSTRUCTIONS</div>
                     )}
                   </div>
                 </div>
                 
                 {/* Action panel */}
                 <div className="panel action-panel">
                   <div className="panel-header"><h2>LAST ACTION & THOUGHT</h2></div>
                   <div className="panel-body">
                     {lastAction ? (
                       <div className="action-content">
                         <div className="action-name">
                           <span className="label">ACTION:</span>
                           <span className="value">{lastAction}</span>
                         </div>
                         {lastActionParams && Object.keys(lastActionParams).length > 0 && (
                           <div className="action-params small-params">
                             <span className="label">PARAMS:</span>
                             <span className="value">{JSON.stringify(lastActionParams)}</span>
                           </div>
                         )}
                         {lastThought && (
                           <div className="action-thought">
                             <span className="label">THOUGHT:</span>
                             <span className="value">{lastThought}</span>
                           </div>
                         )}
                       </div>
                     ) : (
                       <div className="no-action status-text">NO ACTIONS RECORDED</div>
                     )}
                   </div>
                 </div>
            </div>

            {/* Column 2: Object Detection & Odometry */} 
            <div className="grid-column column-2">
                 {/* Object Detection Display Component - Takes significant space */} 
                 <ObjectDetectionDisplay objectDetection={objectDetection} />
                 
                 {/* Odometry panel */}
                 <div className="panel odometry-panel">
                   <div className="panel-header"><h2>POSITION & ORIENTATION</h2></div>
                   <div className="panel-body">
                     {odometry ? (
                       <div className="odometry-content compact-odometry"> {/* Add class for compact display */} 
                         <div className="position-data">
                           <span className="label">POS (m):</span> 
                           X: {formatNumber(odometry.position?.x)} | 
                           Y: {formatNumber(odometry.position?.y)} | 
                           Z: {formatNumber(odometry.position?.z)}
                         </div>
                         <div className="orientation-data">
                           <span className="label">ORIENT (°):</span> 
                           R: {formatNumber(odometry.orientation?.roll)} | 
                           P: {formatNumber(odometry.orientation?.pitch)} | 
                           Y: {formatNumber(odometry.orientation?.yaw)}
                         </div>
                       </div>
                     ) : (
                       <div className="no-odometry status-text waiting">Awaiting odometry...</div>
                     )}
                   </div>
                 </div>
            </div>
            
         </div> 
         
         {/* Removed Tab-specific content for vision, motion, debug */}
         
      </main>

      {/* Footer can remain simple */}
      {/* 
      <footer className="dashboard-footer">
          <p>&copy; {new Date().getFullYear()} SpotAgent Interface</p>
      </footer> 
      */}
    </div>
  );
};

// Error Screen Component
const ErrorScreen = () => (
  <div className="error-screen">
    <h1>ERROR</h1>
    <p>System initialization failed</p>
    <Link to="/" className="retry-btn">RETRY</Link>
  </div>
);

// Main App Component with Routing
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<SplashScreen />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/error" element={<ErrorScreen />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App; 