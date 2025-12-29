/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'index',

    {
      type: 'category',
      label: 'ACT I — AWAKENING INTELLIGENCE',
      items: [
        'chapter-1-embodiment-awakening',
        'chapter-2-sensor-mapping',
        'chapter-3-capability-analysis',
      ],
    },

    {
      type: 'category',
      label: 'ACT II — THE NERVOUS SYSTEM',
      items: [
        'chapter-4-ros-graph',
        'chapter-5-first-motion',
        'chapter-6-robot-skeleton',
      ],
    },

    {
      type: 'category',
      label: 'ACT III — THE DIGITAL TWIN',
      items: [
        'chapter-7-physics-simulation',
        'chapter-8-sensor-awareness',
        'chapter-9-hri-scene',
      ],
    },

    {
      type: 'category',
      label: 'ACT IV — THE ROBOT BRAIN',
      items: [
        'chapter-10-synthetic-vision',
        'chapter-11-navigation-slam',
        'chapter-12-learning-to-move',
      ],
    },

    {
      type: 'category',
      label: 'ACT V — WHEN ROBOTS UNDERSTAND US',
      items: [
        'chapter-13-voice-command',
        'chapter-14-language-model',
        'chapter-15-vla-system',
      ],
    },

    {
      type: 'category',
      label: 'ACT VI — BUILDING THE AUTONOMOUS ROBOT',
      items: [
        'chapter-16-system-integration',
        'chapter-17-sim-to-real',
        'chapter-18-final-demo',
      ],
    },

    {
      type: 'category',
      label: 'ACT VII — RESPONSIBILITY & FUTURE',
      items: [
        'chapter-19-safety-audit',
        'chapter-20-personal-roadmap',
      ],
    },
  ],
};

export default sidebars;
