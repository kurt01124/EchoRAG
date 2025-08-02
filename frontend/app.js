class RAGChatApp {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.isLoading = false;
        this.messageHistory = [];
        
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
        
        // 통계 요소들
        this.serverStatus = document.getElementById('serverStatus');
        this.modelStatus = document.getElementById('modelStatus');
        this.docCount = document.getElementById('docCount');
        this.queryCount = document.getElementById('queryCount');
        this.avgResponse = document.getElementById('avgResponse');
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
                stats: data.stats
            });

            // 통계 업데이트
            this.updateStats(data.stats, data.timing);

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

    async startStatsPolling() {
        // 5초마다 서버 상태 확인
        setInterval(async () => {
            try {
                const response = await fetch(`${this.apiUrl}/stats`);
                const stats = await response.json();
                
                if (stats.averages) {
                    this.queryCount.textContent = stats.averages.total_queries || 0;
                    this.avgResponse.textContent = stats.averages.avg_gpt_time || '-';
                }
                
                this.docCount.textContent = stats.document_count?.toLocaleString() || 0;
                
            } catch (error) {
                console.error('통계 업데이트 실패:', error);
            }
        }, 5000);
    }

    // 유틸리티 메서드들
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
            totalMessages: this.messageHistory.length
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

// 키보드 단축키
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
});

// 앱 초기화
document.addEventListener('DOMContentLoaded', () => {
    window.ragChat = new RAGChatApp();
    
    // 개발자 콘솔에 도움말 출력
    console.log(`
🚀 RAG Chat AI 시스템
=====================
단축키:
- Ctrl+K: 메모리 초기화
- Ctrl+M: 메모리 상태 확인  
- Ctrl+E: 채팅 히스토리 내보내기

API 엔드포인트:
- GET /health: 서버 상태
- POST /chat: 채팅 메시지
- GET /memory: 메모리 상태
- DELETE /memory: 메모리 초기화
- GET /stats: 성능 통계
    `);
});