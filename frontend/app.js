class RAGChatApp {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.isLoading = false;
        this.messageHistory = [];
        this.adminModeEnabled = false; // 🆕 관리자 모드 상태
        
        this.initializeElements();
        this.bindEvents();
        this.checkServerHealth();
        this.startStatsPolling();
    }

    initializeElements() {
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.statusIndicator = document.getElementById('statusIndicator');
        
        // 기본 통계 요소들
        this.serverStatus = document.getElementById('serverStatus');
        this.modelStatus = document.getElementById('modelStatus');
        this.docCount = document.getElementById('docCount');
        this.queryCount = document.getElementById('queryCount');
        this.avgResponse = document.getElementById('avgResponse');
        
        // 🆕 관리자 모드 요소들
        this.adminToggle = document.getElementById('adminToggle');
        this.mlopsPanel = document.getElementById('mlopsPanel');
        this.statsPanel = document.getElementById('statsPanel');
        
        // 🆕 MLOps 통계 요소들
        this.totalCollected = document.getElementById('totalCollected');
        this.newDataCount = document.getElementById('newDataCount');
        this.trainingProgress = document.getElementById('trainingProgress');
        this.progressFill = document.getElementById('progressFill');
        this.trainingStatus = document.getElementById('trainingStatus');
        this.modelVersion = document.getElementById('modelVersion');
        this.batchSize = document.getElementById('batchSize');
        this.pendingCount = document.getElementById('pendingCount');
    }

    bindEvents() {
        // 전송 버튼 클릭
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // 엔터키로 메시지 전송
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // 입력 필드 포커스 시 스크롤
        this.chatInput.addEventListener('focus', () => {
            setTimeout(() => this.scrollToBottom(), 300);
        });
        
        // 🆕 관리자 모드 토글
        this.adminToggle.addEventListener('click', () => this.toggleAdminMode());
    }

    // 🆕 관리자 모드 토글 기능
    toggleAdminMode() {
        this.adminModeEnabled = !this.adminModeEnabled;
        
        if (this.adminModeEnabled) {
            this.mlopsPanel.classList.add('visible');
            this.statsPanel.classList.add('expanded');
            this.adminToggle.classList.add('active');
            console.log('🔧 관리자 모드 활성화');
            
            // MLOps 데이터 즉시 업데이트
            this.updateMLOpsData();
        } else {
            this.mlopsPanel.classList.remove('visible');
            this.statsPanel.classList.remove('expanded');
            this.adminToggle.classList.remove('active');
            console.log('👤 사용자 모드 활성화');
        }
    }

    async checkServerHealth() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const health = await response.json();
            
            this.updateServerStatus(health);
            console.log('서버 상태:', health);
            
        } catch (error) {
            console.error('서버 연결 실패:', error);
            this.updateServerStatus(null);
        }
    }

    updateServerStatus(health) {
        if (health) {
            this.serverStatus.textContent = health.status;
            this.modelStatus.textContent = health.model_loaded ? '로딩됨' : '로딩 중';
            this.docCount.textContent = health.document_count.toLocaleString();
            
            // 상태 인디케이터 업데이트
            if (health.status === 'healthy') {
                this.statusIndicator.style.background = '#4CAF50';
            } else {
                this.statusIndicator.style.background = '#FF9800';
            }
        } else {
            this.serverStatus.textContent = '연결 실패';
            this.modelStatus.textContent = '알 수 없음';
            this.statusIndicator.style.background = '#F44336';
        }
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        
        if (!message || this.isLoading) {
            return;
        }

        // UI 업데이트
        this.addMessage(message, 'user');
        this.chatInput.value = '';
        this.setLoading(true);

        try {
            // API 호출
            const response = await fetch(`${this.apiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    user_id: 'web_user'
                })
            });

            if (!response.ok) {
                throw new Error(`서버 오류: ${response.status}`);
            }

            const data = await response.json();
            
            // 응답 메시지 추가
            this.addMessage(data.response, 'assistant', {
                searchResults: data.search_results,
                timing: data.timing,
                stats: data.stats,
                mlopsInfo: data.mlops_info // 🆕 MLOps 정보 추가
            });

            // 통계 업데이트
            this.updateStats(data.stats, data.timing);
            
            // 🆕 관리자 모드에서 MLOps 정보 업데이트
            if (this.adminModeEnabled && data.mlops_info) {
                this.updateMLOpsInfo(data.mlops_info);
            }

        } catch (error) {
            console.error('메시지 전송 실패:', error);
            this.addMessage(
                `죄송합니다. 오류가 발생했습니다: ${error.message}`, 
                'assistant', 
                { isError: true }
            );
        } finally {
            this.setLoading(false);
        }
    }

    addMessage(content, role, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const avatarIcon = role === 'user' ? 'fas fa-user' : 'fas fa-robot';
        
        let messageHTML = `
            <div class="message-avatar">
                <i class="${avatarIcon}"></i>
            </div>
            <div class="message-content">
                <div class="message-text">${this.formatMessageContent(content)}</div>
        `;

        // 검색 결과 표시
        if (metadata.searchResults && metadata.searchResults.length > 0) {
            messageHTML += `
                <div class="search-results">
                    <strong>🔍 참고한 문서:</strong>
                    ${metadata.searchResults.map(result => `
                        <div class="search-result-item">
                            <small>Score: ${result.score.toFixed(4)}</small><br>
                            ${this.truncateText(result.document, 100)}
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // 타이밍 정보 표시
        if (metadata.timing) {
            messageHTML += `
                <div class="message-meta">
                    ⏱️ 응답시간: ${metadata.timing.total} | 검색: ${metadata.timing.search} | GPT: ${metadata.timing.gpt}
                </div>
            `;
        }

        // 🆕 관리자 모드에서 MLOps 정보 표시
        if (this.adminModeEnabled && metadata.mlopsInfo && role === 'assistant') {
            messageHTML += this.formatMLOpsInfo(metadata.mlopsInfo);
        }

        // 오류 메시지 스타일
        if (metadata.isError) {
            messageHTML += `</div>`;
            messageDiv.innerHTML = messageHTML;
            messageDiv.querySelector('.message-content').classList.add('error-message');
        } else {
            messageHTML += `</div>`;
            messageDiv.innerHTML = messageHTML;
        }

        // 메시지 추가 및 애니메이션
        this.chatMessages.appendChild(messageDiv);
        this.messageHistory.push({ content, role, timestamp: new Date(), metadata });
        
        // 환영 메시지 제거
        const welcomeMessage = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        this.scrollToBottom();
    }

    // 🆕 MLOps 정보 포맷팅
    formatMLOpsInfo(mlopsInfo) {
        if (!mlopsInfo || !mlopsInfo.collected) return '';
        
        const statusIcon = mlopsInfo.training_triggered ? '🚀' : 
                          mlopsInfo.training_queued ? '⏳' : 
                          mlopsInfo.should_train ? '⚡' : '📊';
        
        const statusText = mlopsInfo.training_triggered ? '학습 시작됨' :
                          mlopsInfo.training_queued ? '학습 대기 중' :
                          mlopsInfo.should_train ? '학습 준비됨' :
                          '데이터 수집 중';
        
        return `
            <div class="message-meta" style="background: rgba(103, 126, 234, 0.1); padding: 8px; border-radius: 5px; margin-top: 8px;">
                ${statusIcon} <strong>MLOps:</strong> ${statusText} 
                (총 ${mlopsInfo.total_collected}개, 신규 ${mlopsInfo.new_data_count || 0}개)
                ${mlopsInfo.current_version ? `| 버전: ${mlopsInfo.current_version}` : ''}
            </div>
        `;
    }

    formatMessageContent(content) {
        // 간단한 마크다운 지원
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    setLoading(loading) {
        this.isLoading = loading;
        this.sendButton.disabled = loading;
        this.chatInput.disabled = loading;
        
        if (loading) {
            this.typingIndicator.style.display = 'flex';
            this.sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        } else {
            this.typingIndicator.style.display = 'none';
            this.sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
        
        this.scrollToBottom();
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }

    updateStats(stats, timing) {
        if (stats) {
            this.queryCount.textContent = stats.total_queries || 0;
            this.avgResponse.textContent = stats.avg_gpt || '-';
        }
    }

    // 🆕 MLOps 정보 업데이트
    updateMLOpsInfo(mlopsInfo) {
        if (!mlopsInfo) return;
        
        // 기본 정보 업데이트
        if (mlopsInfo.total_collected !== undefined) {
            this.totalCollected.textContent = `${mlopsInfo.total_collected}개`;
        }
        
        if (mlopsInfo.new_data_count !== undefined) {
            this.newDataCount.textContent = `${mlopsInfo.new_data_count}개`;
        }
        
        // 진행률 계산 및 업데이트
        if (mlopsInfo.pending_count !== undefined) {
            const batchSize = 50; // 기본값, 실제로는 서버에서 받아와야 함
            const progress = Math.max(0, 100 - (mlopsInfo.pending_count / batchSize * 100));
            this.trainingProgress.textContent = `${Math.round(progress)}%`;
            this.progressFill.style.width = `${progress}%`;
            this.pendingCount.textContent = mlopsInfo.pending_count > 0 ? `${mlopsInfo.pending_count}개` : '준비됨';
        }
        
        // 학습 상태 업데이트
        let statusClass = 'waiting';
        let statusText = '대기';
        
        if (mlopsInfo.training_triggered) {
            statusClass = 'training';
            statusText = '학습 중';
        } else if (mlopsInfo.training_queued) {
            statusClass = 'queued';
            statusText = '대기열';
        } else if (mlopsInfo.should_train) {
            statusClass = 'waiting';
            statusText = '준비됨';
        }
        
        this.trainingStatus.className = `status-badge ${statusClass}`;
        this.trainingStatus.textContent = statusText;
        
        // 모델 버전 업데이트
        if (mlopsInfo.current_version) {
            this.modelVersion.textContent = mlopsInfo.current_version;
        }
    }

    // 🆕 MLOps 데이터 별도 조회
    async updateMLOpsData() {
        if (!this.adminModeEnabled) return;
        
        try {
            const response = await fetch(`${this.apiUrl}/mlops/training-progress`);
            const data = await response.json();
            
            // 배치 크기 업데이트
            if (data.batch_size) {
                this.batchSize.textContent = `${data.batch_size}개`;
            }
            
            // 진행률 업데이트
            const progress = data.progress_percentage || 0;
            this.trainingProgress.textContent = `${Math.round(progress)}%`;
            this.progressFill.style.width = `${progress}%`;
            
            // 기타 정보 업데이트
            this.totalCollected.textContent = `${data.current_conversations || 0}개`;
            this.pendingCount.textContent = data.conversations_until_training > 0 ? 
                `${data.conversations_until_training}개` : '준비됨';
            
            // 학습 상태
            let statusClass = 'waiting';
            let statusText = '대기';
            
            if (data.training_in_progress) {
                statusClass = 'training';
                statusText = '학습 중';
            } else if (data.conversations_until_training === 0) {
                statusClass = 'waiting';
                statusText = '준비됨';
            }
            
            this.trainingStatus.className = `status-badge ${statusClass}`;
            this.trainingStatus.textContent = statusText;
            
            // 모델 버전
            if (data.current_version) {
                this.modelVersion.textContent = data.current_version;
            }
            
        } catch (error) {
            console.error('MLOps 데이터 업데이트 실패:', error);
        }
    }

    async startStatsPolling() {
        // 5초마다 서버 상태 확인
        setInterval(async () => {
            try {
                const response = await fetch(`${this.apiUrl}/stats`);
                const stats = await response.json();
                
                if (stats.performance && stats.performance.averages) {
                    this.queryCount.textContent = stats.performance.averages.total_queries || 0;
                    this.avgResponse.textContent = stats.performance.averages.avg_gpt_time || '-';
                }
                
                if (stats.performance) {
                    this.docCount.textContent = stats.performance.document_count?.toLocaleString() || 0;
                }
                
                // 🆕 관리자 모드에서 MLOps 데이터도 업데이트
                if (this.adminModeEnabled) {
                    this.updateMLOpsData();
                }
                
            } catch (error) {
                console.error('통계 업데이트 실패:', error);
            }
        }, 5000);
    }

    // 🆕 관리자 기능들
    async triggerManualTraining() {
        if (!this.adminModeEnabled) return;
        
        try {
            const response = await fetch(`${this.apiUrl}/mlops/finetune`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    force: true,
                    backup_existing: true
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addMessage(
                    `🚀 수동 파인튜닝이 시작되었습니다. (${result.training_data_count}개 데이터)`,
                    'assistant'
                );
            } else {
                this.addMessage(
                    `❌ 파인튜닝 시작 실패: ${result.message}`,
                    'assistant'
                );
            }
            
        } catch (error) {
            console.error('수동 파인튜닝 실패:', error);
            this.addMessage('❌ 파인튜닝 요청 중 오류가 발생했습니다.', 'assistant');
        }
    }

    async clearMLOpsData() {
        if (!this.adminModeEnabled) return;
        
        if (!confirm('수집된 대화 데이터를 모두 삭제하시겠습니까? (백업이 생성됩니다)')) {
            return;
        }
        
        try {
            const response = await fetch(`${this.apiUrl}/mlops/conversations?backup=true`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addMessage(
                    `🗑️ 대화 데이터가 초기화되었습니다. (${result.cleared_count}개 삭제)`,
                    'assistant'
                );
                this.updateMLOpsData(); // 즉시 업데이트
            } else {
                this.addMessage('❌ 데이터 초기화 실패', 'assistant');
            }
            
        } catch (error) {
            console.error('데이터 초기화 실패:', error);
            this.addMessage('❌ 데이터 초기화 중 오류가 발생했습니다.', 'assistant');
        }
    }

    // 기존 유틸리티 메서드들
    async clearMemory() {
        try {
            const response = await fetch(`${this.apiUrl}/memory`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                this.addMessage('메모리가 초기화되었습니다.', 'assistant');
            }
        } catch (error) {
            console.error('메모리 초기화 실패:', error);
        }
    }

    async getMemoryStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/memory`);
            const memory = await response.json();
            
            this.addMessage(
                `현재 메모리: ${memory.count}/${memory.max_count} 메시지`, 
                'assistant'
            );
        } catch (error) {
            console.error('메모리 상태 조회 실패:', error);
        }
    }

    exportChatHistory() {
        const chatData = {
            timestamp: new Date().toISOString(),
            messages: this.messageHistory,
            totalMessages: this.messageHistory.length,
            adminMode: this.adminModeEnabled
        };

        const blob = new Blob([JSON.stringify(chatData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat-history-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// 키보드 단축키 (확장됨)
document.addEventListener('keydown', (e) => {
    // Ctrl+K: 메모리 초기화
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        if (window.ragChat) {
            window.ragChat.clearMemory();
        }
    }
    
    // Ctrl+M: 메모리 상태 확인
    if (e.ctrlKey && e.key === 'm') {
        e.preventDefault();
        if (window.ragChat) {
            window.ragChat.getMemoryStatus();
        }
    }
    
    // Ctrl+E: 채팅 히스토리 내보내기
    if (e.ctrlKey && e.key === 'e') {
        e.preventDefault();
        if (window.ragChat) {
            window.ragChat.exportChatHistory();
        }
    }
    
// 키보드 단축키 (브라우저 충돌 방지)
document.addEventListener('keydown', (e) => {
    // Ctrl+K: 메모리 초기화
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        if (window.ragChat) {
            window.ragChat.clearMemory();
        }
    }
    
    // Ctrl+M: 메모리 상태 확인
    if (e.ctrlKey && e.key === 'm') {
        e.preventDefault();
        if (window.ragChat) {
            window.ragChat.getMemoryStatus();
        }
    }
    
    // Ctrl+E: 채팅 히스토리 내보내기
    if (e.ctrlKey && e.key === 'e') {
        e.preventDefault();
        if (window.ragChat) {
            window.ragChat.exportChatHistory();
        }
    }
    
    // 🆕 Ctrl+Shift+A: 관리자 모드 토글 (충돌 방지)
    if (e.ctrlKey && e.shiftKey && e.key === 'A') {
        e.preventDefault();
        if (window.ragChat) {
            window.ragChat.toggleAdminMode();
        }
    }
    
    // 🆕 Ctrl+Shift+F: 수동 파인튜닝 (Finetune 의 F)
    if (e.ctrlKey && e.shiftKey && e.key === 'F') {
        e.preventDefault();
        if (window.ragChat && window.ragChat.adminModeEnabled) {
            window.ragChat.triggerManualTraining();
        }
    }
    
    // 🆕 Ctrl+Shift+R: MLOps 데이터 초기화 (Reset 의 R)
    if (e.ctrlKey && e.shiftKey && e.key === 'R') {
        e.preventDefault();
        if (window.ragChat && window.ragChat.adminModeEnabled) {
            window.ragChat.clearMLOpsData();
        }
    }
    
    // 🆕 Alt+M: MLOps 상태 새로고침 (관리자 모드에서만)
    if (e.altKey && e.key === 'm') {
        e.preventDefault();
        if (window.ragChat && window.ragChat.adminModeEnabled) {
            window.ragChat.updateMLOpsData();
        }
    }
});
});

// 앱 초기화
document.addEventListener('DOMContentLoaded', () => {
    window.ragChat = new RAGChatApp();
    
    // 개발자 콘솔에 도움말 출력 (확장됨)
    console.log(`
🚀 RAG Chat AI 시스템 (MLOps 지원)
=======================================
기본 단축키:
- Ctrl+K: 메모리 초기화
- Ctrl+M: 메모리 상태 확인  
- Ctrl+E: 채팅 히스토리 내보내기

관리자 모드 단축키:
- Ctrl+Shift+A: 관리자 모드 토글
- Ctrl+Shift+F: 수동 파인튜닝 트리거
- Ctrl+Shift+R: MLOps 데이터 초기화
- Alt+M: MLOps 상태 새로고침

API 엔드포인트:
- GET /health: 서버 상태
- POST /chat: 채팅 메시지
- GET /memory: 메모리 상태
- DELETE /memory: 메모리 초기화
- GET /stats: 성능 통계

MLOps API:
- GET /mlops/status: MLOps 전체 상태
- GET /mlops/training-progress: 학습 진행 상황
- POST /mlops/finetune: 수동 파인튜닝
- DELETE /mlops/conversations: 데이터 초기화
=======================================
💡 Tip: 브라우저 단축키 충돌을 피하기 위해 Ctrl+Shift 조합을 사용합니다!
    `);
});