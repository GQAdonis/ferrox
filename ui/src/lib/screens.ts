import { Activity, Boxes, MessagesSquare, Plug } from "lucide-react";

/**
 * The four screens, in the order the sidebar lists them.
 *
 * One list, read by the sidebar and by the router, so a screen cannot
 * exist in the nav and nowhere else — or the reverse.
 */
export const SCREENS = [
  {
    to: "/ui/chat",
    label: "Chat",
    icon: MessagesSquare,
    blurb: "Stream a completion",
  },
  {
    to: "/ui/models",
    label: "Models",
    icon: Boxes,
    blurb: "Load and download",
  },
  {
    to: "/ui/activity",
    label: "Activity",
    icon: Activity,
    blurb: "Live request log",
  },
  {
    to: "/ui/connect",
    label: "Connect",
    icon: Plug,
    blurb: "Point a tool here",
  },
] as const;
