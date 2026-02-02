use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub type SessionId = String;

#[derive(Debug, Clone)]
pub struct SessionState {
    pub kv_cache_handle: Option<u64>,
    pub last_used: Instant,
    pub history: Vec<(String, String)>,
    pub current_pos: usize,
    pub has_context: bool,
    pub engine_id: Option<usize>,
    pub inflight: bool,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionGcEvent {
    pub session_id: SessionId,
    pub engine_id: Option<usize>,
}

pub trait SessionStore {
    fn get_or_create(&self, session_id: SessionId) -> SessionState;
    fn touch(&self, session_id: &SessionId);
    fn release(&self, session_id: &SessionId);
    fn gc(&self);
}

#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    inner: Mutex<HashMap<SessionId, SessionState>>,
    ttl: Option<Duration>,
    max_sessions: Option<usize>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self::new_with_limits(None, None)
    }

    pub fn new_with_limits(ttl: Option<Duration>, max_sessions: Option<usize>) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            ttl,
            max_sessions,
        }
    }
}

impl SessionStore for InMemorySessionStore {
    fn get_or_create(&self, session_id: SessionId) -> SessionState {
        let mut inner = self.inner.lock().expect("session store mutex poisoned");
        let now = Instant::now();
        inner
            .entry(session_id)
            .or_insert_with(|| SessionState {
                kv_cache_handle: None,
                last_used: now,
                history: Vec::new(),
                current_pos: 0,
                has_context: false,
                engine_id: None,
                inflight: false,
                system_prompt: None,
            })
            .clone()
    }

    fn touch(&self, session_id: &SessionId) {
        let mut inner = self.inner.lock().expect("session store mutex poisoned");
        if let Some(state) = inner.get_mut(session_id) {
            state.last_used = Instant::now();
        }
    }

    fn release(&self, session_id: &SessionId) {
        let mut inner = self.inner.lock().expect("session store mutex poisoned");
        inner.remove(session_id);
    }

    fn gc(&self) {
        let _ = self.gc_collect();
    }
}

impl InMemorySessionStore {
    pub fn gc_collect(&self) -> Vec<SessionGcEvent> {
        let mut inner = self.inner.lock().expect("session store mutex poisoned");
        let now = Instant::now();
        let mut evicted = Vec::new();

        if let Some(ttl) = self.ttl {
            let mut expired = Vec::new();
            for (session_id, state) in inner.iter() {
                if !state.inflight && now.duration_since(state.last_used) > ttl {
                    expired.push(session_id.clone());
                }
            }
            for session_id in expired {
                if let Some(state) = inner.remove(&session_id) {
                    evicted.push(SessionGcEvent {
                        session_id,
                        engine_id: state.engine_id,
                    });
                }
            }
        }

        if let Some(max_sessions) = self.max_sessions {
            while inner.len() > max_sessions {
                let lru_id = inner
                    .iter()
                    .filter(|(_, state)| !state.inflight)
                    .min_by_key(|(_, state)| state.last_used)
                    .map(|(session_id, _)| session_id.clone());
                let Some(session_id) = lru_id else {
                    break;
                };
                if let Some(state) = inner.remove(&session_id) {
                    evicted.push(SessionGcEvent {
                        session_id,
                        engine_id: state.engine_id,
                    });
                }
            }
        }

        evicted
    }

    pub fn get_state(&self, session_id: &SessionId) -> Option<SessionState> {
        let inner = self.inner.lock().expect("session store mutex poisoned");
        inner.get(session_id).cloned()
    }

    pub fn try_begin(&self, session_id: &SessionId) -> bool {
        let mut inner = self.inner.lock().expect("session store mutex poisoned");
        let now = Instant::now();
        let state = inner
            .entry(session_id.clone())
            .or_insert_with(|| SessionState {
                kv_cache_handle: None,
                last_used: now,
                history: Vec::new(),
                current_pos: 0,
                has_context: false,
                engine_id: None,
                inflight: false,
                system_prompt: None,
            });
        if state.inflight {
            return false;
        }
        state.inflight = true;
        state.last_used = Instant::now();
        true
    }

    pub fn end(&self, session_id: &SessionId) {
        let mut inner = self.inner.lock().expect("session store mutex poisoned");
        if let Some(state) = inner.get_mut(session_id) {
            state.inflight = false;
            state.last_used = Instant::now();
        }
    }

    pub fn update_state<F>(&self, session_id: &SessionId, f: F)
    where
        F: FnOnce(&mut SessionState),
    {
        let mut inner = self.inner.lock().expect("session store mutex poisoned");
        let now = Instant::now();
        let state = inner
            .entry(session_id.clone())
            .or_insert_with(|| SessionState {
                kv_cache_handle: None,
                last_used: now,
                history: Vec::new(),
                current_pos: 0,
                has_context: false,
                engine_id: None,
                inflight: false,
                system_prompt: None,
            });
        f(state);
        state.last_used = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gc_ttl_evicts_idle_sessions() {
        let store = InMemorySessionStore::new_with_limits(Some(Duration::from_millis(10)), None);
        let id = "s1".to_string();
        let _ = store.get_or_create(id.clone());
        store.update_state(&id, |state| {
            state.engine_id = Some(3);
            state.inflight = false;
        });
        std::thread::sleep(Duration::from_millis(20));

        let evicted = store.gc_collect();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].session_id, id);
        assert_eq!(evicted[0].engine_id, Some(3));
        assert!(store.get_state(&"s1".to_string()).is_none());
    }

    #[test]
    fn gc_ttl_keeps_inflight_sessions() {
        let store = InMemorySessionStore::new_with_limits(Some(Duration::from_millis(10)), None);
        let id = "busy".to_string();
        let _ = store.get_or_create(id.clone());
        store.update_state(&id, |state| {
            state.inflight = true;
        });
        std::thread::sleep(Duration::from_millis(20));

        let evicted = store.gc_collect();
        assert!(evicted.is_empty());
        assert!(store.get_state(&id).is_some());
    }

    #[test]
    fn gc_lru_evicts_oldest_non_inflight() {
        let store = InMemorySessionStore::new_with_limits(None, Some(2));
        let a = "a".to_string();
        let b = "b".to_string();
        let c = "c".to_string();
        let _ = store.get_or_create(a.clone());
        let _ = store.get_or_create(b.clone());
        let _ = store.get_or_create(c.clone());

        store.update_state(&a, |state| {
            state.last_used = Instant::now() - Duration::from_secs(3);
            state.inflight = false;
        });
        store.update_state(&b, |state| {
            state.last_used = Instant::now() - Duration::from_secs(2);
            state.inflight = false;
        });
        store.update_state(&c, |state| {
            state.last_used = Instant::now() - Duration::from_secs(1);
            state.inflight = false;
        });

        let evicted = store.gc_collect();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].session_id, a);
        assert!(store.get_state(&b).is_some());
        assert!(store.get_state(&c).is_some());
    }

    #[test]
    fn gc_lru_skips_inflight_sessions() {
        let store = InMemorySessionStore::new_with_limits(None, Some(1));
        let a = "a".to_string();
        let b = "b".to_string();
        let _ = store.get_or_create(a.clone());
        let _ = store.get_or_create(b.clone());

        store.update_state(&a, |state| {
            state.last_used = Instant::now() - Duration::from_secs(2);
            state.inflight = true;
        });
        store.update_state(&b, |state| {
            state.last_used = Instant::now() - Duration::from_secs(1);
            state.inflight = false;
        });

        let evicted = store.gc_collect();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].session_id, b);
        assert!(store.get_state(&a).is_some());
        assert!(store.get_state(&"b".to_string()).is_none());
    }
}
