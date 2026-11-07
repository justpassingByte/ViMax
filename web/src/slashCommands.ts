export type SlashCommand = {
  name: string;
  description: string;
};

export type SlashCommandMatch = SlashCommand & {
  matchedPrefix: string;
  unmatchedSuffix: string;
};

export function getSlashCommands(locale: 'vi' | 'en' = 'vi'): SlashCommand[] {
  return [
    {
      name: '/compact',
      description: locale === 'vi' ? 'Nén và tối ưu hóa ngữ cảnh của dự án hiện tại' : 'Compact the current session context',
    },
  ];
}

export function matchingSlashCommands(input: string, locale: 'vi' | 'en' = 'vi'): SlashCommandMatch[] {
  if (!input.startsWith('/')) return [];
  const query = input.trimStart().split(/\s+/, 1)[0] ?? '';
  return getSlashCommands(locale)
    .filter((command) => command.name.toLowerCase().startsWith(query.toLowerCase()))
    .map((command) => ({
      ...command,
      matchedPrefix: command.name.slice(0, query.length),
      unmatchedSuffix: command.name.slice(query.length),
    }));
}

export function shouldShowSlashCommands(input: string, busy: boolean) {
  return !busy && input.startsWith('/');
}
