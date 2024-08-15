'use client';

import { AgentPublic } from '@/cohere-client';
import { IconButton } from '@/components/IconButton';
import { ShareModal } from '@/components/ShareModal';
import { Button, Icon, Logo, Text } from '@/components/Shared';
import { useContextStore } from '@/context';
import { env } from '@/env.mjs';
import { useBrandedColors } from '@/hooks/brandedColors';
import { useConversationStore, useSettingsStore } from '@/stores';
import { cn } from '@/utils';

type Props = {
  agent?: AgentPublic;
};

export const Header: React.FC<Props> = ({ agent }) => {
  const {
    conversation: { id },
  } = useConversationStore();
  const { setLeftPanelOpen, setRightPanelOpen } = useSettingsStore();
  const { open } = useContextStore();
  const { text, bg, contrastText, lightText, fill, lightFill, dark, light } = useBrandedColors(
    agent?.id
  );

  const handleOpenShareModal = () => {
    if (!id) return;
    open({
      title: 'Share link to conversation',
      content: <ShareModal conversationId={id} />,
    });
  };

  const handleOpenLeftSidePanel = () => {
    setRightPanelOpen(false);
    setLeftPanelOpen(true);
  };

  const handleOpenRightSidePanel = () => {
    setLeftPanelOpen(false);
    setRightPanelOpen(true);
  };

  return (
    <div className="flex h-header w-full min-w-0 items-center">
      <div className="flex w-full flex-1 items-center justify-between px-6">
        <div className="flex items-center gap-4">
          <button
            onClick={handleOpenLeftSidePanel}
            className="flex h-full items-center gap-4 lg:hidden"
          >
            {agent ? (
              <Text
                className={cn(
                  'size-5 rounded text-center align-middle uppercase',
                  bg,
                  contrastText
                )}
                styleAs="p"
              >
                {agent?.name[0]}
              </Text>
            ) : (
              <Logo hasCustomLogo={env.NEXT_PUBLIC_HAS_CUSTOM_LOGO} includeBrandName={false} />
            )}
            <Text className="truncate dark:text-mushroom-950" styleAs="p-lg" as="span">
              {agent?.name ?? 'Cohere AI'}
            </Text>
          </button>
          {agent && (
            <Text
              styleAs="label-sm"
              className={cn(
                'rounded bg-mushroom-950 px-2  py-1 font-bold uppercase dark:bg-volcanic-200',
                text,
                dark(lightText)
              )}
            >
              {agent.is_private ? 'Private' : 'Public'}
            </Text>
          )}
        </div>
        <section className="flex items-center gap-4">
          {id && (
            <Button
              kind="secondary"
              className="[&>div]:gap-x-0 lg:[&>div]:gap-x-3"
              label={
                <Text className={cn(dark(lightText), light(text), 'hidden lg:flex')}>Share</Text>
              }
              iconOptions={{
                customIcon: (
                  <Icon name="share" kind="outline" className={cn(light(fill), dark(lightFill))} />
                ),
              }}
              iconPosition="start"
              onClick={handleOpenShareModal}
            />
          )}
          <IconButton
            iconName="kebab"
            iconClassName="dark:fill-marble-950 fill-volcanic-100"
            onClick={handleOpenRightSidePanel}
            className="flex h-auto w-auto lg:hidden"
          />
        </section>
      </div>
    </div>
  );
};
