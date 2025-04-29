import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Helper function to create a camera frustum visual group (unposed)
function createCameraFrustumGroupUnposed(name, cameraData, color = 0x00ffff) {
  const { intrinsics, dimensions } = cameraData;

  // Calculate FOV and aspect ratio
  const fovY = 2 * Math.atan(dimensions.height / (2 * intrinsics.focal_length_y)) * (180 / Math.PI);
  const aspect = dimensions.width / dimensions.height;
  const near = 0.02; // Use a smaller near plane for the helper camera
  const far = 0.05;  // Far plane distance for visualization (Keep reduced size)

  // Create a camera at the origin to define the frustum shape for the helper
  const helperCam = new THREE.PerspectiveCamera(fovY, aspect, near, far);

  // Create the visual helper
  const helper = new THREE.CameraHelper(helperCam);
  helper.material.color.setHex(color);
  helper.material.linewidth = 2;

  // Create the Group that will hold the helper
  const cameraGroup = new THREE.Group();
  cameraGroup.userData.name = name; // Store name on the group
  cameraGroup.add(helper);
  cameraGroup.matrixAutoUpdate = false; // We will set the world matrix directly

  return cameraGroup; // Return the unposed group
}

const ThreeDVisualization = ({ objectDetection, odometry }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const robotOriginRef = useRef(null); // Represents the robot's body frame origin
  const cameraFrustumsContainerRef = useRef(null); // Container for frustum groups in world space
  const objectsGroupRef = useRef(null);
  const rayGroupRef = useRef(null);

  useEffect(() => {
    const currentMount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(75, currentMount.clientWidth / currentMount.clientHeight, 0.1, 1000);
    camera.position.set(2, 1.5, 2);
    camera.lookAt(0, 0.5, 0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(currentMount.clientWidth, currentMount.clientHeight);
    rendererRef.current = renderer;
    currentMount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 0.5;
    controls.maxDistance = 20;
    controls.target.set(0, 0.5, 0);
    controlsRef.current = controls;

    const gridHelper = new THREE.GridHelper(10, 10);
    gridHelper.position.y = 0;
    scene.add(gridHelper);

    const worldAxesHelper = new THREE.AxesHelper(1);
    scene.add(worldAxesHelper);

    // Group to represent the robot's body frame origin and orientation
    const robotOriginGroup = new THREE.Group();
    robotOriginRef.current = robotOriginGroup;
    scene.add(robotOriginGroup); // Add to scene to get world matrix
    
    const robotAxesHelper = new THREE.AxesHelper(0.5);
    robotOriginGroup.add(robotAxesHelper);

    // *** Apply horizontal flip using scale ***
    robotOriginGroup.scale.x = -1;

    // Group for camera frustums (PARENTED TO ROBOT ORIGIN)
    const cameraFrustumsGroup = new THREE.Group();
    cameraFrustumsContainerRef.current = cameraFrustumsGroup;
    robotOriginGroup.add(cameraFrustumsGroup);

    // Group for detected objects (PARENTED TO ROBOT ORIGIN)
    const objectsGroup = new THREE.Group();
    objectsGroupRef.current = objectsGroup;
    robotOriginGroup.add(objectsGroup); // Parent objects to robot

    // Group for ray cast visualization (PARENTED TO ROBOT ORIGIN)
    const rayGroup = new THREE.Group();
    rayGroupRef.current = rayGroup;
    robotOriginGroup.add(rayGroup); // Parent rays to robot

    const ambientLight = new THREE.AmbientLight(0x606060);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
        if (currentMount) {
            const width = currentMount.clientWidth;
            const height = currentMount.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        }
    };
    handleResize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (currentMount && rendererRef.current.domElement && currentMount.contains(rendererRef.current.domElement)) {
         currentMount.removeChild(rendererRef.current.domElement);
      }
      renderer.dispose();
      // Dispose scene children geometries/materials if needed
    };
  }, []);

  // Update robot origin pose
  useEffect(() => {
    if (odometry && robotOriginRef.current) {
      const pos = odometry.position || { x: 0, y: 0, z: 0 };
      const ori = odometry.orientation || { yaw: 0 };
      robotOriginRef.current.position.set(pos.x, pos.z, pos.y);
      robotOriginRef.current.rotation.set(0, THREE.MathUtils.degToRad(ori.yaw), 0);
      robotOriginRef.current.updateMatrixWorld(true); // Ensure world matrix is up-to-date
      if (controlsRef.current) {
           controlsRef.current.target.copy(robotOriginRef.current.position).add(new THREE.Vector3(0, 0.5, 0));
      }
    }
  }, [odometry]);

  // Update camera frustums using LOCAL pose relative to robotOrigin
  useEffect(() => {
      const robotOrigin = robotOriginRef.current;
      const frustumsContainer = cameraFrustumsContainerRef.current;
      if (objectDetection?.camera_details && robotOrigin && frustumsContainer) {
          const cameraDetails = objectDetection.camera_details;
          const existingFrustumGroups = frustumsContainer.children.reduce((acc, child) => {
              if (child.userData.name) acc[child.userData.name] = child;
              return acc;
          }, {});
          const currentFrustumNames = new Set();
          // const scaleVec = new THREE.Vector3(1, 1, 1); // Not needed for local pose

          // Add/Update frustum groups using LOCAL pose relative to robotOrigin
          for (const cameraName in cameraDetails) {
              currentFrustumNames.add(cameraName);
              const cameraData = cameraDetails[cameraName];
              const pose = cameraData.pose;

              // LOCAL pose relative to robotOrigin (X, Z, Y mapping for position)
              const positionVecRel = new THREE.Vector3(pose.position.x, pose.position.z, pose.position.y);
              const quaternionRel = new THREE.Quaternion(pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w);
              // *** Reinstate fixed adjustment for coord system alignment ***
              const qAdjustX = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2); // Align Z up with Y up (+90 deg)
              const qAdjustZ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI);    // Flip 180 deg around original forward
              const qAdjustCombined = new THREE.Quaternion().multiplyQuaternions(qAdjustX, qAdjustZ);
              quaternionRel.premultiply(qAdjustCombined);
              
              let frustumGroup = existingFrustumGroups[cameraName];
              if (!frustumGroup) {
                  frustumGroup = createCameraFrustumGroupUnposed(cameraName, cameraData);
                  frustumsContainer.add(frustumGroup);
              } 
              // Apply LOCAL pose directly
              frustumGroup.position.copy(positionVecRel);
              frustumGroup.quaternion.copy(quaternionRel);
              frustumGroup.matrixAutoUpdate = true; // Use auto matrix updates now
          }

          // Remove old frustum groups
          for (const name in existingFrustumGroups) {
              if (!currentFrustumNames.has(name)) {
                  const groupToRemove = existingFrustumGroups[name];
                  frustumsContainer.remove(groupToRemove);
              }
          }
      }
  }, [objectDetection?.camera_details]);

  // Update detected objects and ray cast (using LOCAL positions relative to robotOrigin)
  useEffect(() => {
    const objectsGroup = objectsGroupRef.current;
    const rayGroup = rayGroupRef.current;
    const frustumsContainer = cameraFrustumsContainerRef.current;

    if (objectDetection?.objects && objectsGroup && rayGroup && frustumsContainer) {

      // --- Cleanup --- 
      while (objectsGroup.children.length > 0) {
          const obj = objectsGroup.children[0];
          objectsGroup.remove(obj);
          if (obj.geometry) obj.geometry.dispose();
          if (obj.material) { if (obj.material.map) obj.material.map.dispose(); obj.material.dispose(); }
      }
      while (rayGroup.children.length > 0) {
          const line = rayGroup.children[0];
          rayGroup.remove(line);
          if (line.geometry) line.geometry.dispose();
          if (line.material) line.material.dispose();
      }
      // --- End Cleanup --- 

      objectDetection.objects.forEach((obj) => {
        if (obj.position && obj.source_camera) {
          // Object position relative to robot origin (X, Z, Y mapping)
          const objLocalPos = new THREE.Vector3(obj.position.x, obj.position.z, obj.position.y);
          
          // Add object sphere at its local position
          const sphereGeometry = new THREE.SphereGeometry(0.1, 8, 8);
          const sphereMaterial = new THREE.MeshPhongMaterial({ color: 0xFF0000 });
          const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
          sphere.position.copy(objLocalPos);
          objectsGroup.add(sphere);

          // Find the corresponding camera pose group
          const cameraPoseGroup = frustumsContainer.children.find(g => g.userData.name === obj.source_camera);
          if (cameraPoseGroup) {
            // Get the camera's position *relative to the robot origin*
            const cameraLocalPos = cameraPoseGroup.position;

            // Add ray cast line (coordinates are relative to robotOriginGroup)
            const points = [ cameraLocalPos, objLocalPos ];
            const rayGeometry = new THREE.BufferGeometry().setFromPoints(points);
            const rayMaterial = new THREE.LineBasicMaterial({ color: 0x00FF00 });
            const rayLine = new THREE.Line(rayGeometry, rayMaterial);
            rayGroup.add(rayLine);
          }

          // Add label (position relative to robot origin)
          const canvas = document.createElement('canvas');
          const context = canvas.getContext('2d');
          canvas.width = 256; canvas.height = 64;
          context.fillStyle = 'rgba(0,0,0,0.7)';
          context.fillRect(0, 0, canvas.width, canvas.height);
          context.font = '32px Arial'; context.fillStyle = 'white';
          context.textAlign = 'center'; context.textBaseline = 'middle';
          context.fillText(obj.label, canvas.width / 2, canvas.height / 2);
          const texture = new THREE.CanvasTexture(canvas);
          texture.needsUpdate = true;
          const spriteMaterial = new THREE.SpriteMaterial({ map: texture, depthTest: false });
          const sprite = new THREE.Sprite(spriteMaterial);
          // Position label slightly above object's local position (using X, Z, Y mapping)
          sprite.position.set(objLocalPos.x, objLocalPos.y + 0.2, objLocalPos.z); 
          sprite.scale.set(0.5, 0.25, 1);
          sprite.renderOrder = 1;
          objectsGroup.add(sprite);
        }
      });
    }
  }, [objectDetection]);

  return (
    <div ref={mountRef} style={{ width: '100%', height: '400px' }} />
  );
};

export default ThreeDVisualization; 