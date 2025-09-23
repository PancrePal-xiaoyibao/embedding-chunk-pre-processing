
// 应用JavaScript功能
document.addEventListener('DOMContentLoaded', function() {
    // 自动刷新处理状态
    if (window.location.pathname.includes('/process/')) {
        const taskId = window.location.pathname.split('/').pop();
        const statusInterval = setInterval(function() {
            fetch(`/api/status/${taskId}`)
                .then(response => response.json())
                .then(data => {
                    updateProcessStatus(data);
                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(statusInterval);
                        if (data.status === 'completed') {
                            setTimeout(() => {
                                window.location.href = `/results/${taskId}`;
                            }, 2000);
                        }
                    }
                })
                .catch(error => {
                    console.error('状态更新失败:', error);
                });
        }, 2000);
    }
    
    // 文件上传进度
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const fileSize = (file.size / 1024 / 1024).toFixed(2);
                const fileInfo = document.createElement('div');
                fileInfo.className = 'alert alert-info mt-2';
                fileInfo.innerHTML = `
                    <i class="bi bi-file-earmark"></i>
                    已选择文件: ${file.name} (${fileSize} MB)
                `;
                
                const existingInfo = document.querySelector('.file-upload-info');
                if (existingInfo) {
                    existingInfo.remove();
                }
                
                fileInfo.className += ' file-upload-info';
                fileInput.parentNode.appendChild(fileInfo);
            }
        });
    }
    
    // 配置表单验证
    const configForm = document.querySelector('form[action*="config"]');
    if (configForm) {
        configForm.addEventListener('submit', function(e) {
            const targetSize = parseInt(document.getElementById('target_chunk_size')?.value || 0);
            const minSize = parseInt(document.getElementById('min_chunk_size')?.value || 0);
            const maxSize = parseInt(document.getElementById('max_chunk_size')?.value || 0);
            
            if (minSize >= targetSize || targetSize >= maxSize) {
                e.preventDefault();
                alert('分块大小配置错误：最小 < 目标 < 最大');
                return false;
            }
        });
    }
});

function updateProcessStatus(data) {
    const progressBar = document.querySelector('.progress-bar');
    const statusBadge = document.querySelector('.status-badge');
    const messageElement = document.querySelector('.status-message');
    
    if (progressBar) {
        progressBar.style.width = data.progress + '%';
        progressBar.setAttribute('aria-valuenow', data.progress);
        progressBar.textContent = data.progress + '%';
    }
    
    if (statusBadge) {
        statusBadge.className = 'badge status-badge';
        switch(data.status) {
            case 'pending':
                statusBadge.className += ' bg-secondary';
                break;
            case 'processing':
                statusBadge.className += ' bg-primary processing-animation';
                break;
            case 'completed':
                statusBadge.className += ' bg-success';
                break;
            case 'failed':
                statusBadge.className += ' bg-danger';
                break;
        }
        statusBadge.textContent = getStatusText(data.status);
    }
    
    if (messageElement) {
        messageElement.textContent = data.message || '';
    }
}

function getStatusText(status) {
    const statusMap = {
        'pending': '等待中',
        'processing': '处理中',
        'completed': '已完成',
        'failed': '失败'
    };
    return statusMap[status] || status;
}

// 工具函数
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }
}
