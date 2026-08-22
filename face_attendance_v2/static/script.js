// Interactive JavaScript for Face Attendance System v2

// Tab Switching on Login Page
function switchTab(tabName) {
  const adminForm = document.getElementById('admin-form');
  const studentForm = document.getElementById('student-form');
  const tabAdmin = document.getElementById('tab-admin');
  const tabStudent = document.getElementById('tab-student');

  if (tabName === 'admin') {
    adminForm.style.display = 'block';
    studentForm.style.display = 'none';
    tabAdmin.classList.add('active');
    tabStudent.classList.remove('active');
  } else {
    adminForm.style.display = 'none';
    studentForm.style.display = 'block';
    tabAdmin.classList.remove('active');
    tabStudent.classList.add('active');
  }
}

// Student Registration Browser Webcam Capture
let webcamStream = null;
let capturedSnapshots = [];

function startWebcam() {
  const video = document.getElementById('webcam-video');
  if (!video) return;

  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      webcamStream = stream;
      video.srcObject = stream;
      video.play();
      document.getElementById('capture-btn').disabled = false;
    })
    .catch((err) => {
      alert("Could not access browser webcam: " + err.message);
    });
}

function captureSnapshot() {
  const video = document.getElementById('webcam-video');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL('image/jpeg');
  capturedSnapshots.push(dataUrl);

  // Update UI count
  const countSpan = document.getElementById('snapshot-count');
  if (countSpan) countSpan.innerText = capturedSnapshots.length;

  // Append hidden input to form
  const snapshotsContainer = document.getElementById('snapshots-container');
  if (snapshotsContainer) {
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = 'snapshots';
    input.value = dataUrl;
    snapshotsContainer.appendChild(input);
  }

  // Display thumbnail preview
  const gallery = document.getElementById('snapshot-gallery');
  if (gallery) {
    const img = document.createElement('img');
    img.src = dataUrl;
    img.className = 'thumbnail-img';
    gallery.appendChild(img);
  }
}
