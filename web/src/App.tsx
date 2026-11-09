import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {
  ArrowUp,
  Brain,
  Braces,
  CircleStop,
  Clock3,
  FilePenLine,
  FileText,
  Film,
  Folder,
  FolderPlus,
  Files,
  Image as ImageIcon,
  Menu,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRight,
  Plus,
  Save,
  Search,
  Settings,
  ListChecks,
  Terminal,
  Trash2,
  Sun,
  Wrench,
  X,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {deleteSession, getAgentConfig, getArtifacts, getHistory, getSessions, saveAgentConfig, sendMessage, startAgent, stopAgent, subscribeToEvents, uploadWorkspaceFile} from './api';
import {ArtifactsView, StoryboardPanel} from './ArtifactViews';
import {translations, resolveLocale, LOCALE_STORAGE_KEY, type Locale} from './i18n';
import {applyAgentEvent, appendLocalUser, composeAgentPrompt, createChatState, humanize} from './events';
import {matchingSlashCommands, shouldShowSlashCommands, type SlashCommandMatch} from './slashCommands';
import {applyTheme, resolveTheme, THEME_STORAGE_KEY, type Theme} from './theme';
import type {AgentConfig, AgentEvent, Artifact, ChatState, ConfigSection, Message, SessionSummary, WorkspaceUpload} from './types';

const CONTEXT_TARGET = 160_000;

type WorkspaceView = 'workspace' | 'artifacts' | 'settings';

export default function App() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState('');
  const [chat, setChat] = useState<ChatState>(() => createChatState());
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>('workspace');
  const [agentReady, setAgentReady] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [storyboardPanelOpen, setStoryboardPanelOpen] = useState(false);
  const [storyboardCount, setStoryboardCount] = useState(0);
  const [theme, setTheme] = useState<Theme>(() => resolveTheme(document.documentElement.dataset.theme, false));
  const [locale, setLocale] = useState<Locale>(() => resolveLocale());
  const t = translations[locale];

  useEffect(() => {
    try {
      window.localStorage.setItem(LOCALE_STORAGE_KEY, locale);
    } catch {}
  }, [locale]);
  const [draft, setDraft] = useState('');
  const [workspaceUploads, setWorkspaceUploads] = useState<WorkspaceUpload[]>([]);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [newProjectOpen, setNewProjectOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectError, setNewProjectError] = useState('');
  const [creatingProject, setCreatingProject] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<SessionSummary>();
  const [deleting, setDeleting] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedSession = sessions.find((session) => session.sessionId === selectedSessionId);
  const slashMatches = useMemo(() => matchingSlashCommands(draft, locale), [draft, locale]);
  const showSlashCommands = shouldShowSlashCommands(draft, chat.busy);
  const runningRender = useMemo(
    () => [...chat.messages].reverse().find((message) => message.role === 'activity'
      && message.status === 'running'
      && (message.tool || '').toLowerCase().includes('render_video')),
    [chat.messages],
  );

  useEffect(() => {
    applyTheme(theme);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Theme still applies when persistence is unavailable.
    }
  }, [theme]);

  const refreshSessions = useCallback(async () => {
    const state = await getSessions();
    setSessions(state.sessions);
    return state;
  }, []);

  const refreshArtifacts = useCallback(async (sessionId: string) => {
    if (!sessionId) {
      setArtifacts([]);
      return;
    }
    const payload = await getArtifacts(sessionId);
    setArtifacts(payload.artifacts);
  }, []);

  useEffect(() => subscribeToEvents((event) => {
    if (event.type === 'sessions_changed') {
      setSessions(event.sessions || []);
      if (event.activeSessionId) setSelectedSessionId(event.activeSessionId);
      return;
    }
    if (event.type === 'bridge_status') {
      if (event.status === 'ready' || event.status === 'starting') setAgentReady(true);
      if (event.status === 'stopped' || event.status === 'error') setAgentReady(false);
    }
    if (event.type === 'session') {
      const sessionId = event.session?.active_session_id || event.session?.session?.session_id || '';
      if (sessionId) {
        setSelectedSessionId(sessionId);
        void refreshSessions();
        void refreshArtifacts(sessionId);
      }
    }
    setChat((current) => applyAgentEvent(current, event));
  }, () => undefined), [refreshArtifacts, refreshSessions]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const state = await refreshSessions();
        if (cancelled || !state.activeSessionId) return;
        setSelectedSessionId(state.activeSessionId);
        const [history] = await Promise.all([
          getHistory(state.activeSessionId),
          refreshArtifacts(state.activeSessionId),
          startAgent({sessionId: state.activeSessionId}),
        ]);
        if (!cancelled) {
          setChat(createChatState(history.messages));
          setAgentReady(true);
        }
      } catch (error) {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : String(error));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshArtifacts, refreshSessions]);

  useEffect(() => {
    if (!runningRender || !selectedSessionId) return;
    void refreshArtifacts(selectedSessionId);
    const interval = window.setInterval(() => void refreshArtifacts(selectedSessionId), 2_000);
    return () => {
      window.clearInterval(interval);
      void refreshArtifacts(selectedSessionId);
    };
  }, [refreshArtifacts, runningRender, selectedSessionId]);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    element.scrollTo({top: element.scrollHeight, behavior: chat.busy ? 'smooth' : 'auto'});
  }, [chat.messages, chat.busy]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = '0px';
    textarea.style.height = `${Math.min(168, Math.max(28, textarea.scrollHeight))}px`;
  }, [draft]);

  useEffect(() => {
    setWorkspaceUploads([]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [selectedSessionId]);

  async function openSession(sessionId: string) {
    if (!sessionId || sessionId === selectedSessionId) {
      setMobileSidebarOpen(false);
      return;
    }
    setLoadError('');
    setSelectedSessionId(sessionId);
    setMobileSidebarOpen(false);
    setChat(createChatState());
    try {
      const [history] = await Promise.all([
        getHistory(sessionId),
        refreshArtifacts(sessionId),
        startAgent({sessionId}),
      ]);
      setChat(createChatState(history.messages));
      setAgentReady(true);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }

  function openNewProjectDialog() {
    setNewProjectName('');
    setNewProjectError('');
    setNewProjectOpen(true);
  }

  async function newProject() {
    const projectName = newProjectName.trim();
    if (!projectName || creatingProject) return;
    setCreatingProject(true);
    setNewProjectError('');
    setLoadError('');
    setMobileSidebarOpen(false);
    try {
      await startAgent({newSession: true, projectName});
      setAgentReady(true);
      await new Promise((resolve) => window.setTimeout(resolve, 450));
      const state = await refreshSessions();
      setSelectedSessionId(state.activeSessionId);
      setChat(createChatState());
      setArtifacts([]);
      setWorkspaceView('workspace');
      setNewProjectOpen(false);
      setNewProjectName('');
      textareaRef.current?.focus();
    } catch (error) {
      setNewProjectError(error instanceof Error ? error.message : String(error));
    } finally {
      setCreatingProject(false);
    }
  }

  async function submit() {
    const text = draft.trim();
    if (!text || chat.busy || uploadingFiles) return;
    setLoadError('');
    setDraft('');
    setChat((current) => appendLocalUser(current, text));
    try {
      if (!agentReady) {
        await startAgent(selectedSessionId ? {sessionId: selectedSessionId} : {newSession: true});
        setAgentReady(true);
      }
      const outbound = composeAgentPrompt(text, workspaceUploads.map((file) => file.path));
      await sendMessage(outbound);
      setWorkspaceUploads([]);
    } catch (error) {
      const event: AgentEvent = {type: 'error', message: error instanceof Error ? error.message : String(error)};
      setChat((current) => applyAgentEvent(current, event));
    }
  }

  async function uploadFiles(files: FileList | null) {
    const sessionId = selectedSessionId;
    if (!files?.length) return;
    if (!sessionId) {
      setLoadError(t.createProjectFirst);
      return;
    }
    setUploadingFiles(true);
    setLoadError('');
    let uploadedAny = false;
    try {
      for (const file of Array.from(files)) {
        const result = await uploadWorkspaceFile(sessionId, file);
        uploadedAny = true;
        setWorkspaceUploads((current) => [
          ...current.filter((item) => item.path !== result.file.path),
          result.file,
        ]);
      }
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
      if (uploadedAny) await refreshArtifacts(sessionId);
      setUploadingFiles(false);
      textareaRef.current?.focus();
    }
  }

  async function stop() {
    await stopAgent();
    setAgentReady(false);
    setChat((current) => ({...current, busy: false}));
  }

  async function confirmDelete() {
    if (!pendingDelete || deleting) return;
    const sessionId = pendingDelete.sessionId;
    const deletingSelected = sessionId === selectedSessionId;
    setDeleting(true);
    setLoadError('');
    try {
      const state = await deleteSession(sessionId);
      setSessions(state.sessions);
      setPendingDelete(undefined);
      if (deletingSelected) {
        setSelectedSessionId('');
        setChat(createChatState());
        setArtifacts([]);
        setAgentReady(false);
        if (state.activeSessionId) await openSession(state.activeSessionId);
      }
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    } finally {
      setDeleting(false);
    }
  }

  const contextPercent = Math.min(100, Math.round((chat.promptTokens / CONTEXT_TARGET) * 100));
  const hasConversation = chat.messages.length > 0;

  return (
    <div className="app-shell">
      <Sidebar
        locale={locale}
        t={t}
        open={sidebarOpen}
        mobileOpen={mobileSidebarOpen}
        sessions={sessions}
        selectedSessionId={selectedSessionId}
        activeView={workspaceView}
        onToggle={() => setSidebarOpen((value) => !value)}
        onMobileClose={() => setMobileSidebarOpen(false)}
        onNew={openNewProjectDialog}
        onSelect={(sessionId) => void openSession(sessionId)}
        onWorkspace={() => {
          setWorkspaceView('workspace');
          setMobileSidebarOpen(false);
        }}
        onArtifacts={() => {
          setWorkspaceView('artifacts');
          setMobileSidebarOpen(false);
        }}
        onSettings={() => {
          setWorkspaceView('settings');
          setMobileSidebarOpen(false);
        }}
        onDelete={setPendingDelete}
      />

      <main className="workspace-main">
        {workspaceView === 'workspace' ? (
          <div className="workspace-utility-bar">
            <button className="icon-button mobile-only" onClick={() => setMobileSidebarOpen(true)} aria-label="Open navigation">
              <Menu size={19} />
            </button>
            {!sidebarOpen && (
              <button className="icon-button desktop-only" onClick={() => setSidebarOpen(true)} aria-label="Open navigation">
                <PanelLeftOpen size={18} />
              </button>
            )}
            <span className="workspace-utility-spacer" />
            <div className="workspace-utility-actions">
              <LocaleToggle locale={locale} onToggle={() => setLocale((cur) => cur === 'vi' ? 'en' : 'vi')} />
              <ThemeToggle theme={theme} onToggle={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} />
              <button className="icon-button artifact-toggle" onClick={() => setStoryboardPanelOpen((value) => !value)} aria-label="Toggle storyboard preview">
                <PanelRight size={18} />
                {storyboardCount > 0 && <span className="count-badge">{storyboardCount}</span>}
              </button>
            </div>
          </div>
        ) : (
          <header className="workspace-header">
            <div className="workspace-title-row">
              <button className="icon-button mobile-only" onClick={() => setMobileSidebarOpen(true)} aria-label="Open navigation">
                <Menu size={19} />
              </button>
              {!sidebarOpen && (
                <button className="icon-button desktop-only" onClick={() => setSidebarOpen(true)} aria-label="Open navigation">
                  <PanelLeftOpen size={18} />
                </button>
              )}
              <div className="workspace-title-copy">
                <strong>{workspaceView === 'settings' ? t.settings : t.artifacts}</strong>
                {workspaceView === 'settings' && <span>configs/agent.local.yaml</span>}
              </div>
            </div>
            <div className="workspace-utility-actions">
              <LocaleToggle locale={locale} onToggle={() => setLocale((cur) => cur === 'vi' ? 'en' : 'vi')} />
              <ThemeToggle theme={theme} onToggle={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} />
            </div>
          </header>
        )}

        {workspaceView === 'workspace' ? (
          <div className="conversation" ref={scrollRef}>
            {!hasConversation ? (
              <EmptyState theme={theme} t={t} />
            ) : (
              <div className="message-stream">
                {chat.messages.map((message) => <MessageRow key={message.id} message={message} />)}
                {chat.busy && <ThinkingRow messages={chat.messages} />}
              </div>
            )}
          </div>
        ) : workspaceView === 'artifacts' ? (
          <ArtifactsView session={selectedSession} artifacts={artifacts} locale={locale} />
        ) : (
          <SettingsView locale={locale} t={t} />
        )}

        {workspaceView === 'workspace' && <div className="composer-zone">
          {loadError && (
            <div className="inline-error" role="alert">
              <span>{loadError}</span>
              <button onClick={() => setLoadError('')} aria-label="Dismiss error"><X size={15} /></button>
            </div>
          )}
          {showSlashCommands && <SlashCommandMenu matches={slashMatches} contextPercent={contextPercent} onSelect={(command) => {
            setDraft(command);
            textareaRef.current?.focus();
          }} />}
          <div className={`composer ${chat.busy ? 'is-busy' : ''}`}>
            <textarea
              ref={textareaRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Tab' && slashMatches[0]) {
                  event.preventDefault();
                  setDraft(slashMatches[0].name);
                  return;
                }
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  void submit();
                }
              }}
              placeholder={t.describeIdea}
              aria-label={t.messageViMax}
              disabled={chat.busy}
              rows={1}
            />
            {(workspaceUploads.length > 0 || uploadingFiles) && (
              <div className="composer-attachments" aria-live="polite">
                {workspaceUploads.map((file) => (
                  <span className="composer-attachment" key={file.path} title={file.path}>
                    <FileText size={13} />
                    <span>{file.name}</span>
                    <button
                      type="button"
                      onClick={() => setWorkspaceUploads((current) => current.filter((item) => item.path !== file.path))}
                      aria-label={`Remove ${file.name} from this message`}
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
                {uploadingFiles && <span className="composer-uploading">{t.uploading}</span>}
              </div>
            )}
            <div className="composer-controls">
              <input
                ref={fileInputRef}
                className="composer-file-input"
                type="file"
                multiple
                onChange={(event) => void uploadFiles(event.currentTarget.files)}
                tabIndex={-1}
              />
              <button
                type="button"
                className="composer-add"
                onClick={() => fileInputRef.current?.click()}
                disabled={!selectedSessionId || uploadingFiles || chat.busy}
                aria-label={t.uploadTooltip}
                aria-busy={uploadingFiles}
                title={selectedSessionId ? t.uploadTooltip : t.createProjectFirst}
              >
                <Plus size={20} />
              </button>
              <div className="composer-spacer" />
              {chat.busy ? (
                <button className="send-button stop" onClick={() => void stop()} aria-label={t.stopGeneration}><CircleStop size={18} /></button>
              ) : (
                <button className="send-button" onClick={() => void submit()} disabled={!draft.trim() || uploadingFiles} aria-label={t.sendMessage}><ArrowUp size={19} /></button>
              )}
            </div>
          </div>
        </div>}
      </main>

      <StoryboardPanel
        open={storyboardPanelOpen && workspaceView === 'workspace'}
        artifacts={artifacts}
        locale={locale}
        activeRenderStage={runningRender?.stage}
        onClose={() => setStoryboardPanelOpen(false)}
        onCountChange={setStoryboardCount}
      />
      <DeleteProjectDialog
        locale={locale}
        t={t}
        session={pendingDelete}
        deleting={deleting}
        onCancel={() => !deleting && setPendingDelete(undefined)}
        onConfirm={() => void confirmDelete()}
      />
      <NewProjectDialog
        locale={locale}
        t={t}
        open={newProjectOpen}
        name={newProjectName}
        error={newProjectError}
        creating={creatingProject}
        onNameChange={(value) => {
          setNewProjectName(value);
          setNewProjectError('');
        }}
        onCancel={() => {
          if (creatingProject) return;
          setNewProjectOpen(false);
          setNewProjectError('');
        }}
        onConfirm={() => void newProject()}
      />
    </div>
  );
}

function Sidebar({locale, t, open, mobileOpen, sessions, selectedSessionId, activeView, onToggle, onMobileClose, onNew, onSelect, onWorkspace, onArtifacts, onSettings, onDelete}: {
  locale: Locale;
  t: (typeof translations)[Locale];
  open: boolean;
  mobileOpen: boolean;
  sessions: SessionSummary[];
  selectedSessionId: string;
  activeView: WorkspaceView;
  onToggle: () => void;
  onMobileClose: () => void;
  onNew: () => void;
  onSelect: (sessionId: string) => void;
  onWorkspace: () => void;
  onArtifacts: () => void;
  onSettings: () => void;
  onDelete: (session: SessionSummary) => void;
}) {
  return (
    <>
      {mobileOpen && <button className="sidebar-scrim" onClick={onMobileClose} aria-label="Close navigation" />}
      <aside className={`sidebar ${open ? 'is-open' : 'is-collapsed'} ${mobileOpen ? 'is-mobile-open' : ''}`}>
        <div className="sidebar-brand">
          <strong>ViMax</strong>
          <button className="icon-button sidebar-collapse desktop-only" onClick={onToggle} aria-label="Collapse navigation">
            <PanelLeftClose size={17} />
          </button>
          <button className="icon-button mobile-only" onClick={onMobileClose} aria-label="Close navigation"><X size={18} /></button>
        </div>
        <nav className="primary-nav" aria-label="Primary navigation">
          <button onClick={onNew}><FolderPlus size={17} /><span>{t.newProject}</span></button>
          <button className={activeView === 'workspace' ? 'is-active' : ''} onClick={onWorkspace}><Folder size={17} /><span>{t.workspace}</span></button>
          <button className={activeView === 'artifacts' ? 'is-active' : ''} onClick={onArtifacts}><Files size={17} /><span>{t.artifacts}</span></button>
          <button className={activeView === 'settings' ? 'is-active' : ''} onClick={onSettings}><Settings size={17} /><span>{t.settings}</span></button>
        </nav>
        <div className="session-section">
          <div className="section-label"><span>{t.projects}</span><span>{sessions.length}</span></div>
          <div className="session-list">
            {sessions.map((session) => (
              <div
                key={session.sessionId}
                className={`session-item ${session.sessionId === selectedSessionId ? 'is-selected' : ''}`}
              >
                <button className="session-open" onClick={() => onSelect(session.sessionId)}>
                  <span className="session-copy">
                    <strong>{sessionTitle(session)}</strong>
                    <small>{relativeTime(session.updatedAt, locale)} · {stageLabel(session.stage, locale)}</small>
                  </span>
                </button>
                <button className="session-delete" onClick={() => onDelete(session)} aria-label={`Delete ${sessionTitle(session)}`} title="Delete project">
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
            {sessions.length === 0 && <span className="empty-list">{t.noProjects}</span>}
          </div>
        </div>
        <div className="sidebar-footer">
          <span className="avatar">V</span>
          <div><strong>{t.localWorkspace}</strong><small>{t.localViMax}</small></div>
        </div>
      </aside>
    </>
  );
}

function EmptyState({theme, t}: {theme: Theme; t: (typeof translations)[Locale]}) {
  return (
    <section className="empty-state">
      <img className="empty-state-logo" src={theme === 'dark' ? '/vimax-light.svg' : '/vimax-dark.svg'} alt="ViMax" />
      <h1>{t.whatShouldWeCreate}</h1>
      <p style={{maxWidth: '560px', margin: '14px auto 0', color: 'var(--color-text-muted, #71717a)', fontSize: '13.5px', lineHeight: 1.55, textAlign: 'center'}}>
        {t.emptyStateTip}
      </p>
    </section>
  );
}


function LocaleToggle({locale, onToggle}: {locale: Locale; onToggle: () => void}) {
  return (
    <button
      type="button"
      className="icon-button locale-toggle"
      onClick={onToggle}
      aria-label={locale === 'vi' ? 'Chuyển sang tiếng Anh' : 'Switch to Vietnamese'}
      title={locale === 'vi' ? 'Tiếng Việt (Bấm để đổi sang English)' : 'English (Click to switch to Tiếng Việt)'}
      style={{fontSize: '11px', fontWeight: 700, padding: '0 6px', letterSpacing: '0.5px'}}
    >
      {locale.toUpperCase()}
    </button>
  );
}

function ThemeToggle({theme, onToggle}: {theme: Theme; onToggle: () => void}) {
  const dark = theme === 'dark';
  return (
    <button
      type="button"
      className="icon-button theme-toggle"
      onClick={onToggle}
      aria-label={dark ? 'Use light mode' : 'Use dark mode'}
      aria-pressed={dark}
      title={dark ? 'Light mode' : 'Dark mode'}
    >
      {dark ? <Sun size={18} /> : <Moon size={18} />}
    </button>
  );
}

function MessageRow({message}: {message: Message}) {
  if (message.role === 'activity') return <ActivityRow message={message} />;
  return (
    <article className={`message-row role-${message.role}`}>
      <div className="message-body">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{a: (props) => <a {...props} target="_blank" rel="noreferrer" />}}>
          {message.text}
        </ReactMarkdown>
      </div>
    </article>
  );
}

function ActivityRow({message}: {message: Message}) {
  const stage = message.stage ? humanize(message.stage) : '';
  const detail = stage.toLowerCase() === message.text.toLowerCase() ? message.text : [stage, message.text].filter(Boolean).join(' · ');
  const toolKind = activityToolKind(message.tool);
  return (
    <div className={`activity-row status-${message.status || 'done'}`}>
      <span className={`activity-indicator tool-${toolKind}`}><ActivityToolIcon tool={message.tool} /></span>
      <div>
        <strong>{humanize(message.tool || 'Workflow')}</strong>
        <span>{detail}</span>
      </div>
    </div>
  );
}

function ActivityToolIcon({tool}: {tool?: string}) {
  const name = (tool || '').toLowerCase();
  const props = {size: 13, strokeWidth: 1.8};
  if (name.includes('narrative_planning') || name.includes('novel_planning')) return <FilePenLine {...props} />;
  if (name.includes('render_video')) return <Film {...props} />;
  if (name === 'view_image' || name.includes('image')) return <ImageIcon {...props} />;
  if (name === 'read_json' || name === 'write_json') return <Braces {...props} />;
  if (name === 'read_file' || name === 'write_file') return <FileText {...props} />;
  if (name === 'list_files' || name === 'glob_files') return <Folder {...props} />;
  if (name === 'search_text') return <Search {...props} />;
  if (name.startsWith('memory_')) return <Brain {...props} />;
  if (name.startsWith('todo_')) return <ListChecks {...props} />;
  if (name === 'run_shell') return <Terminal {...props} />;
  if (name === 'sleep') return <Clock3 {...props} />;
  return <Wrench {...props} />;
}

function activityToolKind(tool?: string) {
  const name = (tool || '').toLowerCase();
  if (name.includes('narrative_planning') || name.includes('novel_planning')) return 'planning';
  if (name.includes('render_video')) return 'render';
  if (name === 'view_image' || name.includes('image')) return 'image';
  if (name.startsWith('memory_')) return 'memory';
  if (name.startsWith('todo_')) return 'todo';
  if (name === 'run_shell') return 'shell';
  if (name === 'sleep') return 'time';
  return 'file';
}

function ThinkingRow({messages}: {messages: Message[]}) {
  const running = [...messages].reverse().find((message) => message.role === 'activity' && message.status === 'running');
  return (
    <div className="thinking-row">
      <span className="thinking-mark"><i /><i /><i /></span>
      <span>{running ? `${humanize(running.tool || 'ViMax')} · ${running.text}` : (running ? undefined : 'ViMax đang xử lý...')}</span>
    </div>
  );
}

function SlashCommandMenu({matches, contextPercent, onSelect}: {matches: SlashCommandMatch[]; contextPercent: number; onSelect: (command: string) => void}) {
  return (
    <div className="slash-command-menu" role="listbox" aria-label="Slash commands">
      {matches.length > 0 ? matches.map((command) => (
        <button key={command.name} role="option" aria-selected="false" onMouseDown={(event) => event.preventDefault()} onClick={() => onSelect(command.name)}>
          <code><span><b>{command.matchedPrefix}</b><span>{command.unmatchedSuffix}</span></span>{command.name === '/compact' && <em>{contextPercent}%</em>}</code>
          <small>{command.description}</small>
        </button>
      )) : <span className="slash-command-empty">No matching commands</span>}
    </div>
  );
}

const CONFIG_SECTIONS: Array<{key: keyof AgentConfig['sections']; title: string; description: string}> = [
  {key: 'llm', title: 'Agent LLM', description: 'Planning, tool selection, and conversation'},
  {key: 'audio', title: 'Voiceover & Audio', description: 'Native audio, ElevenLabs, or Edge-TTS narration & dialogue'},
  {key: 'subtitles', title: 'Subtitles', description: 'SRT/ASS subtitles and burn-in preferences'},
  {key: 'image', title: 'Image generation', description: 'Characters, keyframes, and shot frames'},
  {key: 'video', title: 'Video generation', description: 'Shot clips and final video'},
  {key: 'embedding', title: 'Embedding', description: 'Optional novel retrieval'},
  {key: 'reranker', title: 'Reranker', description: 'Optional novel retrieval ranking'},
];

function SettingsView({locale, t}: {locale: Locale; t: (typeof translations)[Locale]}) {
  const [config, setConfig] = useState<AgentConfig>();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState('');

  useEffect(() => {
    let cancelled = false;
    void getAgentConfig()
      .then((payload) => !cancelled && setConfig(payload))
      .catch((error) => !cancelled && setStatus(error instanceof Error ? error.message : String(error)))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, []);

  function update(section: keyof AgentConfig['sections'], field: keyof ConfigSection, value: string | boolean) {
    setStatus('');
    setConfig((current) => {
      if (!current) return current;
      const existing = current.sections[section] || {};
      return {
        sections: {
          ...current.sections,
          [section]: {...existing, [field]: value},
        },
      };
    });
  }

  async function save() {
    if (!config || saving) return;
    setSaving(true);
    setStatus('');
    try {
      setConfig(await saveAgentConfig(config));
      setStatus(t.saved);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <div className="settings-loading">{t.loadingConfig}</div>;
  if (!config) return <div className="settings-loading is-error">{status || t.configUnavailable}</div>;
  return (
    <section className="settings-view">
      <header>
        <div><span>{t.localConfig}</span><h1>{t.settingsTitle}</h1></div>
        <div className="settings-save-group">
          {status && <span className={(status === 'Saved' || status === t.saved) ? 'is-saved' : 'is-error'}>{status}</span>}
          <button className="settings-save" onClick={() => void save()} disabled={saving}>
            <Save size={15} />{saving ? t.saving : t.saveConfig}
          </button>
        </div>
      </header>
      <div className="settings-sections">
        {CONFIG_SECTIONS.map((definition) => (
          <ConfigSectionEditor
            key={definition.key}
            sectionKey={definition.key}
            definition={{
              title: definition.key === 'llm' ? t.sectionLlmTitle : definition.key === 'audio' ? t.sectionAudioTitle : definition.key === 'subtitles' ? t.sectionSubtitlesTitle : definition.key === 'image' ? t.sectionImageTitle : definition.key === 'video' ? t.sectionVideoTitle : definition.key === 'embedding' ? t.sectionEmbeddingTitle : t.sectionRerankerTitle,
              description: definition.key === 'llm' ? t.sectionLlmDesc : definition.key === 'audio' ? t.sectionAudioDesc : definition.key === 'subtitles' ? t.sectionSubtitlesDesc : definition.key === 'image' ? t.sectionImageDesc : definition.key === 'video' ? t.sectionVideoDesc : definition.key === 'embedding' ? t.sectionEmbeddingDesc : t.sectionRerankerDesc,
            }}
            value={config.sections[definition.key]}
            onChange={(field, value) => update(definition.key, field, value)}
            t={t}
          />
        ))}
      </div>
    </section>
  );
}

function ConfigSectionEditor({sectionKey, definition, value, onChange, t}: {
  sectionKey?: keyof AgentConfig['sections'];
  t: (typeof translations)[Locale];
  definition: {title: string; description: string};
  value?: ConfigSection;
  onChange: (field: keyof ConfigSection, value: string | boolean) => void;
}) {
  const current = value || {model: '', base_url: '', api_key: '', has_api_key: false};
  if (sectionKey === 'subtitles') {
    return (
      <section className="config-section">
        <header><h2>{definition.title}</h2><p>{definition.description}</p></header>
        <div className="config-fields">
          <label className="config-toggle config-field-wide">
            <input type="checkbox" checked={Boolean(current.enabled)} onChange={(event) => onChange('enabled', event.target.checked)} />
            <span>{t.subtitlesEnabledLabel}</span>
          </label>
          <label className="config-toggle config-field-wide">
            <input type="checkbox" checked={current.burn_in !== false} onChange={(event) => onChange('burn_in', event.target.checked)} />
            <span>{t.subtitlesBurnInLabel}</span>
          </label>
          <label>
            <span>{t.subtitlesLanguageLabel}</span>
            <input value={current.language ?? 'vi'} onChange={(event) => onChange('language', event.target.value)} placeholder="vi" />
          </label>
          <label>
            <span>{t.subtitlesFontLabel}</span>
            <input value={current.font ?? 'Arial'} onChange={(event) => onChange('font', event.target.value)} placeholder="Arial" />
          </label>
          <label>
            <span>{t.subtitlesFontSizeLabel}</span>
            <input type="number" min="12" max="120" value={current.font_size ?? '34'} onChange={(event) => onChange('font_size', event.target.value)} placeholder="34" />
          </label>
        </div>
      </section>
    );
  }
  return (
    <section className="config-section">
      <header><h2>{definition.title}</h2><p>{definition.description}</p></header>
      <div className="config-fields">
        {current.model_provider !== undefined && (
          <label><span>{t.modelProvider}</span><input value={current.model_provider} onChange={(event) => onChange('model_provider', event.target.value)} /></label>
        )}
        <label><span>{t.model}</span><input value={current.model ?? ''} onChange={(event) => onChange('model', event.target.value)} /></label>
        <label className="config-field-wide"><span>{t.baseUrl}</span><input value={current.base_url ?? ''} onChange={(event) => onChange('base_url', event.target.value)} inputMode="url" /></label>
        <label className="config-field-wide">
          <span>{t.apiKey} <i className={current.has_api_key ? 'is-configured' : ''}>{current.has_api_key ? t.configured : t.notConfigured}</i></span>
          <input type="password" value={current.api_key ?? ''} onChange={(event) => onChange('api_key', event.target.value)} placeholder={current.has_api_key ? t.apiKeyPlaceholderConfigured : t.apiKeyPlaceholderEmpty} autoComplete="off" />
        </label>
        {sectionKey === 'audio' && (
          <>
            <label className="config-field-wide">
              <span>{t.voiceLabel}</span>
              <input value={current.voice ?? ''} onChange={(event) => onChange('voice', event.target.value)} placeholder={t.voicePlaceholder} />
            </label>
            <label className="config-field-wide">
              <span>{t.characterVoicesLabel}</span>
              <textarea
                rows={4}
                value={current.character_voices_text ?? ''}
                onChange={(event) => onChange('character_voices_text', event.target.value)}
                placeholder={t.characterVoicesPlaceholder}
              />
            </label>
            <label>
              <span>{t.bgmFileLabel}</span>
              <input value={current.bgm_file ?? ''} onChange={(event) => onChange('bgm_file', event.target.value)} placeholder={t.bgmFilePlaceholder} />
            </label>
            <label>
              <span>{t.bgmVolumeLabel}</span>
              <input type="number" step="0.05" min="0" max="1" value={current.bgm_volume ?? ''} onChange={(event) => onChange('bgm_volume', event.target.value)} placeholder="0.2" />
            </label>
          </>
        )}
      </div>
    </section>
  );
}

function DeleteProjectDialog({locale, t, session, deleting, onCancel, onConfirm}: {
  locale: Locale;
  t: (typeof translations)[Locale];
  session?: SessionSummary;
  deleting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    if (!session) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !deleting) onCancel();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [deleting, onCancel, session]);

  if (!session) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <section className="delete-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-project-title">
        <span className="dialog-icon"><Trash2 size={18} /></span>
        <div className="dialog-copy">
          <h2 id="delete-project-title">{t.deleteProjectDialogTitle}</h2>
          <p><strong>{sessionTitle(session)}</strong> {t.deleteProjectDialogDesc}</p>
        </div>
        <div className="dialog-actions">
          <button onClick={onCancel} disabled={deleting} autoFocus>{t.cancelBtn}</button>
          <button className="danger" onClick={onConfirm} disabled={deleting}>{deleting ? t.deletingBtn : t.deleteBtn}</button>
        </div>
      </section>
    </div>
  );
}

function NewProjectDialog({locale, t, open, name, error, creating, onNameChange, onCancel, onConfirm}: {
  locale: Locale;
  t: (typeof translations)[Locale];
  open: boolean;
  name: string;
  error: string;
  creating: boolean;
  onNameChange: (value: string) => void;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !creating) onCancel();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [creating, onCancel, open]);

  if (!open) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <form className="project-dialog" role="dialog" aria-modal="true" aria-labelledby="new-project-title" onSubmit={(event) => {
        event.preventDefault();
        onConfirm();
      }}>
        <span className="dialog-icon is-create"><FolderPlus size={18} /></span>
        <div className="dialog-copy">
          <h2 id="new-project-title">{t.createProjectDialogTitle}</h2>
          <p>{t.createProjectDialogDesc}</p>
        </div>
        <label className="project-name-field">
          <span>{t.projectNameLabel}</span>
          <input
            value={name}
            onChange={(event) => onNameChange(event.target.value)}
            placeholder={t.projectNamePlaceholder}
            maxLength={64}
            autoFocus
            disabled={creating}
          />
          {error && <small role="alert">{error}</small>}
        </label>
        <div className="dialog-actions">
          <button type="button" onClick={onCancel} disabled={creating}>{t.cancelBtn}</button>
          <button type="submit" className="primary" disabled={creating || !name.trim()}>{creating ? t.creatingBtn : t.createBtn}</button>
        </div>
      </form>
    </div>
  );
}

function sessionTitle(session?: SessionSummary) {
  if (!session) return 'New video';
  if (session.projectName) return session.projectName;
  const source = session.idea || session.summary;
  if (source) return source.length > 38 ? `${source.slice(0, 38).trim()}…` : source;
  return session.sessionId.replace(/^\d{8}-\d{6}-?/, '') || 'Untitled video';
}

function stageLabel(stage: string, locale: Locale = 'vi') {
  const t = translations[locale];
  const labels: Record<string, string> = {
    created: t.stageCreated,
    narrative_planning: t.stagePlanning,
    narrative_planned: t.stagePlanReady,
    novel_planning: t.stagePlanningNovel,
    novel_planned: t.stageNovelReady,
    rendering: t.stageRendering,
    rendered: t.stageRendered,
    error: t.stageError,
  };
  return labels[stage] || humanize(stage || 'Created');
}

function relativeTime(value: string, locale: Locale = 'vi') {
  const t = translations[locale];
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return t.recently;
  const delta = Math.max(0, Date.now() - timestamp);
  const minutes = Math.floor(delta / 60_000);
  if (minutes < 1) return t.now;
  if (minutes < 60) return `${minutes}${locale === 'vi' ? ' ' + t.minutesAgo : 'm'}`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}${locale === 'vi' ? ' ' + t.hoursAgo : 'h'}`;
  return `${Math.floor(hours / 24)}${locale === 'vi' ? ' ' + t.daysAgo : 'd'}`;
}
