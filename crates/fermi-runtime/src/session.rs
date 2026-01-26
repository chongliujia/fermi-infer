use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

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

pub trait SessionStore {
    fn get_or_create(&self, session_id: SessionId) -> SessionState;
    fn touch(&self, session_id: &SessionId);
    fn release(&self, session_id: &SessionId);
    fn gc(&self);
}

#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    inner: Mutex<HashMap<SessionId, SessionState>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
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
        // Placeholder: implement TTL/LRU cleanup later.
    }
}

impl InMemorySessionStore {
    pub fn get_state(&self, session_id: &SessionId) -> Option<SessionState> {
        let inner = self.inner.lock().expect("session store mutex poisoned");
        inner.get(session_id).cloned()
    }

    pub fn try_begin(&self, session_id: &SessionId) -> bool {
        let mut inner = self.inner.lock().expect("session store mutex poisoned");
        let now = Instant::now();
        let state = inner.entry(session_id.clone()).or_insert_with(|| SessionState {
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
        let state = inner.entry(session_id.clone()).or_insert_with(|| SessionState {
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
